import logging
from typing import Any, Callable, List, Optional, Tuple, Union

from numpy.typing import NDArray
from pixelinkWrapper import*
from ctypes import*
import ctypes.wintypes
import threading
import numpy as np
from qtpy import QtCore
import json
import os

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class PixelinkCamera:
    """Control a Imaging Source camera in the style of pylablib.

    It wraps an :class:`pylon.InstantCamera` instance.

    :param name: Full name of the device.
    :param callback: Callback method for each grabbed image
    """

    # One-time operation states
    INACTIVE = 1    # A one-time operation is NOT in progress
    INPROGRESS = 2  # A one-time operation IS in progress
    STOPPING = 3    # A one-time operation IS in progress, but an abort request has been made.

    def __init__(self, info: str, callback: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)

        self.name = info["Name"]
        self.device_info = info
        self.attributes = {}
        self.open()

        # register device lost event handler
        #ret = PxLApi.setEventCallback(self.camera, PxLApi.EventId.CAMERA_DISCONNECTED, self.name, camera_lost)
        #if not PxLApi.apiSuccess(ret[0]):
        #    print("ERROR setting event callback function: {0}".format(ret[0]))
        #    print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)

        # Callback setup for image grabbing
        self.listener = Listener()
        ret = PxLApi.setCallback(self.camera, PxLApi.Callback.FRAME, self.listener._user_data, self.listener._callback_func)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR setting frame callback function: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)

        # Initialize feature map with current values
        self.feature_map = self.build_feature_param_name_map()
        
        if callback is not None:
            self.set_callback(callback=callback)

    def open(self) -> None:
        # Initialize camera using serial number
        ret = PxLApi.initialize(int(self.device_info["Serial Number"]))
        if PxLApi.apiSuccess(ret[0]):
            self.camera = ret[1]
        self.get_attributes()
        self.attribute_names = [attr['name'] for attr in self.attributes] + [child['name'] for attr in self.attributes if attr.get('type') == 'group' for child in attr.get('children', [])]

    def set_callback(
        self, callback: Callable[[NDArray], None], replace_all: bool = True
    ) -> None:
        """Setup a callback method for continuous acquisition.

        :param callback: Method to be used in continuous mode. It should accept an array as input.
        :param bool replace_all: Whether to remove all previously set callback methods.
        """
        if replace_all:
            try:
                self.listener.signals.data_ready.disconnect()
            except TypeError:
                pass  # not connected
        self.listener.signals.data_ready.connect(callback)
    
    def get_attributes(self):
        """Get the attributes of the camera and store them in a dictionary."""
        name = self.name.replace(" ", "-")
        file_path = os.path.join(os.environ.get('PROGRAMDATA'), '.pymodaq', f'config_{name}.json')
        with open(file_path, 'r') as file:
            attributes = json.load(file)
            self.attributes = self.clean_device_attributes(attributes)

    def get_detector_size(self) -> Tuple[int, int]:
        """Return width and height of detector in pixels."""
        return self.camera.device_property_map['WidthMax'], self.camera.device_property_map['HeightMax']

    def start_acquisition(self) -> None:
        ret = PxLApi.setStreamState(self.camera, PxLApi.StreamState.START)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR setting stream state function: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)

    def stop_acquisition(self) -> None:
        ret = PxLApi.setStreamState(self.camera, PxLApi.StreamState.STOP)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR setting stream state function: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)

    def close(self) -> None:
        ret = PxLApi.setCallback(self.camera, PxLApi.Callback.FRAME, self.listener._user_data, 0)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR disabling frame callback function: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)
        #ret = PxLApi.setEventCallback(self.camera, PxLApi.EventId.ANY, self.name, 0)
        #if not PxLApi.apiSuccess(ret[0]):
        #    print("ERROR disabling event callback function: {0}".format(ret[0]))
        #    print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)
        ret = PxLApi.setStreamState(self.camera, PxLApi.PreviewState.STOP)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR setting preview state function: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)
        ret = PxLApi.setStreamState(self.camera, PxLApi.StreamState.STOP)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR setting stream state function: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)
        ret = PxLApi.uninitialize(self.camera)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR uninitializing camera: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)    

    def save_device_state(self):
        ret = PxLApi.saveSettings(self.camera, PxLApi.Settings.SETTINGS_USER)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR saving device state to non-volatile memory: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)

    def load_device_state(self):
        ret = PxLApi.loadSettings(self.camera, PxLApi.Settings.SETTINGS_USER)
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR loading device state from non-volatile memory: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReport)

    def start_grabbing(self, frame_rate: int) -> None:
        """Start continuously to grab data.

        Whenever a grab succeeded, the callback defined in :meth:`set_callback` is called.
        """
        try:
            #self.camera.device_property_map.set_value(ic4.PropId.ACQUISITION_FRAME_RATE, frame_rate)
            pass
        except Exception:
            pass
        self.start_acquisition()

    def build_feature_param_name_map(self):
        feature_param_name_map = {}

        # Get actual features from camera
        ret = PxLApi.getCameraFeatures(self.camera, PxLApi.FeatureId.ALL)
        if not PxLApi.apiSuccess(ret[0]):
            raise RuntimeError("Failed to get camera features")

        cameraFeatures = ret[1]

        for i in range(cameraFeatures.uNumberOfFeatures):
            feature = cameraFeatures.Features[i]
            feature_id = feature.uFeatureId
            num_params = feature.uNumberOfParameters

            # Get string name of the feature
            feature_name = next(
                (name for name, value in vars(PxLApi.FeatureId).items() if value == feature_id),
                f"FEATURE_{feature_id}"
            ).upper()

            # Try to find associated Params enum
            param_class_name = feature_name.title().replace('_', '') + 'Params'
            param_class = getattr(PxLApi, param_class_name, None)

            if param_class:
                # Named indices
                param_map = {
                    k.upper(): v for k, v in vars(param_class).items()
                    if not k.startswith('__') and isinstance(v, int)
                }
            elif num_params == 1:
                # Use "VALUE" as default parameter name
                param_map = {
                    "VALUE": 0
                }
            else:
                # Fallback to generic naming
                param_map = {
                    f"PARAM_{idx}": idx for idx in range(num_params)
                }

            # Manually add entries for auto exposure continuously and once
            if feature_name == "SHUTTER":
                feature_param_name_map["SHUTTER_AUTO"] = {
                    "id": feature_id,
                    "params": {
                        "VALUE": 0,
                        "AUTO_MIN": 1,
                        "AUTO_MAX": 2
                        }
                }
                feature_param_name_map["SHUTTER_AUTO_ONCE"] = {
                    "id": feature_id,
                    "params": {
                        "VALUE": 0,
                        "AUTO_MIN": 1,
                        "AUTO_MAX": 2
                        }
                }

            # Add structured entry
            feature_param_name_map[feature_name] = {
                "id": feature_id,
                "params": param_map
            }
        


        return feature_param_name_map


    def clean_device_attributes(self, attributes):
        clean_params = []

        # Check if attributes is a list or dictionary
        if isinstance(attributes, dict):
            items = attributes.items()
        elif isinstance(attributes, list):
            # If it's a list, we assume each item is a parameter (no keys)
            items = enumerate(attributes)  # Use index for 'key'
        else:
            raise ValueError(f"Unsupported type for attributes: {type(attributes)}")

        for idx, attr in items:
            param = {}

            param['title'] = attr.get('title', '')
            param['name'] = attr.get('name', str(idx))  # use index if name is missing
            param['type'] = attr.get('type', 'str')
            param['value'] = attr.get('value', '')
            param['default'] = attr.get('default', None)
            param['limits'] = attr.get('limits', None)
            param['readonly'] = attr.get('readonly', False)

            if param['type'] == 'group' and 'children' in attr:
                children = attr['children']
                # If children is a dict, convert to a list
                if isinstance(children, dict):
                    children = list(children.values())
                param['children'] = self.clean_device_attributes(children)

            clean_params.append(param)

        return clean_params
    
    """
    Create a new one-time autoexposure thread
    """
    def create_autoexposure_thread(self):
        return threading.Thread(target=self.perform_one_time_autoexposure, args=(self.camera,), daemon=True)

    """
    One-time autoexposure thread function to perform a one-time autoexposure operation, ending when the operation is complete, 
    or until told to abort.
    """
    def perform_one_time_autoexposure(self):

        global oneTimeAutoExposure
        oneTimeAutoExposure = self.INPROGRESS
        print("\n\nStarting one-time autoexposure adjustment.")

        exposure = 0 # Intialize exposure to 0, but this value is ignored when initating auto adjustment.
        params = [exposure] 

        ret = PxLApi.setFeature(self.camera, PxLApi.FeatureId.EXPOSURE, PxLApi.FeatureFlags.ONEPUSH, params)
        if not(PxLApi.apiSuccess(ret[0])):
            print("!! Attempt to set one-time autoexposure returned %i!" % ret[0])
            oneTimeAutoExposure == self.INACTIVE
            return

        # Now that we have initiated a one-time operation, loop until it is done (or told to abort).
        while oneTimeAutoExposure == self.INPROGRESS:
            ret = PxLApi.getFeature(self.camera, PxLApi.FeatureId.EXPOSURE)
            if PxLApi.apiSuccess(ret[0]):
                flags = ret[1]
                params = ret[2]
                if not (flags & PxLApi.FeatureFlags.ONEPUSH): # the operation completed
                    break
            QtCore.QThread.msleep(20) # Give some time for the one-time operation to complete

        if oneTimeAutoExposure == self.STOPPING:
            ret = PxLApi.setFeature(self.camera, PxLApi.FeatureId.EXPOSURE, PxLApi.FeatureFlags.MANUAL, params)
            if not(PxLApi.apiSuccess(ret[0])):
                print("!! Attempt to aborted one-time autoexposure returned %i!\n"
                    "It will be aborted when it is completed by the camera." % (ret[0]))
                oneTimeAutoExposure == self.INACTIVE
                return
            print("\nFinished one-time autoexposure adjustment. Operation aborted.\n")
        else:
            print("\nFinished one-time autoexposure adjustment. Operation completed successfully.\n")

        oneTimeAutoExposure = self.INACTIVE

        return

    """
    If a one-time autoexposure is not already in progress, starts a thread that will perfrom the
    one-time autoexposure, exiting when it has completed (or is aborted).
    """
    def initiate_one_time_autoexposure(self):
        
        global oneTimeAutoExposure
        if oneTimeAutoExposure == self.INACTIVE:
            oneTimeThread = None
            oneTimeThread = self.create_autoexposure_thread(self.camera)
            if None == oneTimeThread:
                print("  !! Could not create a one-time autoexposure thread!")
            else:
                oneTimeThread.start()
        return
    
    """
    Controls continuous autoexposure
    """
    def set_continuous_autoexposure(self, on):

        exposure = 0 # Intialize exposure to 0, but this value is ignored when initating auto adjustment.
        params = [exposure]

        if not on:
            # We are looking to turn off the continual auto exposure, which means we will be manually
            # adjusting it from now on -- including with the below PxLApi.setFeature. So given that we have to
            # set it to someting, read the current value (as set by the camera), and use that value.
            ret = PxLApi.getFeature(self.camera, PxLApi.FeatureId.EXPOSURE)
            if not PxLApi.apiSuccess(ret[0]):
                print("!! Attempt to get exposure returned %i!" % ret[0])
                return
            params = ret[2]
            flags = PxLApi.FeatureFlags.MANUAL
        else:
            flags = PxLApi.FeatureFlags.AUTO

        ret = PxLApi.setFeature(self.camera, PxLApi.FeatureId.EXPOSURE, flags, params)
        if not PxLApi.apiSuccess(ret[0]):
            print("!! Attempt to set continuous autoexposure returned %i!" % ret[0])
            return

class Listener:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals = self.ListenerSignal()
        self.frame_ready = False

        # Store function pointer for use with API
        # IMPORTANT: Access it only after class definition
        self._callback_func = self.frame_callback
        # IMPORTANT: Must pass a ctypes.c_void_p to setCallback, but you need to point it to a py_object, not to the pointer to that py_object
        self._user_data_obj = ctypes.py_object(self)
        self._user_data = ctypes.cast(ctypes.pointer(self._user_data_obj), ctypes.c_void_p)
    
    @staticmethod
    @PxLApi._dataProcessFunction
    def frame_callback(hCamera, frameData, dataFormat, frameDesc, userData):

        # IMPORTANT: Must cast from void* to POINTER(py_object), then dereference
        self = ctypes.cast(userData, ctypes.POINTER(ctypes.py_object)).contents.value

        frameDescriptor = frameDesc.contents
        width = int(frameDescriptor.Roi.fWidth / frameDescriptor.PixelAddressingValue.fHorizontal)
        height = int(frameDescriptor.Roi.fHeight / frameDescriptor.PixelAddressingValue.fVertical)
        bytesPerPixel = PxLApi.getBytesPerPixel(dataFormat)

        # Emit signal safely
        npFrame = self.numPy_image(frameData, width, height, bytesPerPixel)
        if npFrame is not None:
            self.signals.data_ready.emit(npFrame)
            self.frame_ready = True

        return 0
    
    @staticmethod
    def numPy_image(frameData, width, height, bytesPerPixel):
        size = width * height * bytesPerPixel
        # Cast frameData to a ctypes array and copy into NumPy array
        buf_type = ctypes.c_ubyte * size
        buf = ctypes.cast(frameData, ctypes.POINTER(buf_type)).contents
        arr = np.frombuffer(buf, dtype=np.uint8).copy()
        return arr.reshape(height, width * bytesPerPixel)

    class ListenerSignal(QtCore.QObject):
        data_ready = QtCore.pyqtSignal(object)


def get_info_for_all_cameras():
    ret = PxLApi.getNumberCameras()
    if PxLApi.apiSuccess(ret[0]):
        cameraIdInfo = ret[1]
        numCameras = len(cameraIdInfo)
        devicesInfo = []
        if 0 < numCameras:
            for i in range(numCameras):
                serialNumber = cameraIdInfo[i].CameraSerialNum
                ret = PxLApi.initialize(serialNumber)
                if PxLApi.apiSuccess(ret[0]):
                    hCamera = ret[1]
                    ret = PxLApi.getCameraInfo(hCamera)
                    if PxLApi.apiSuccess(ret[0]):
                        cameraInfo = ret[1]
                        devicesInfo.append(get_camera_info(cameraInfo))
                    PxLApi.uninitialize(hCamera)
        return devicesInfo


def get_camera_info(cameraInfo):
    """
    Get all the info for the camera as a dictionary
    """
    info = {
        "Name": cameraInfo.CameraName.decode("utf-8"),
        "Description": cameraInfo.Description.decode("utf-8"),
        "Vendor Name": cameraInfo.VendorName.decode("utf-8"),
        "Serial Number": cameraInfo.SerialNumber.decode("utf-8"),
        "Firmware Version": cameraInfo.FirmwareVersion.decode("utf-8"),
        "FPGA Version": cameraInfo.FPGAVersion.decode("utf-8"),
        "XML Version": cameraInfo.XMLVersion.decode("utf-8"),
        "Bootload Version": cameraInfo.BootloadVersion.decode("utf-8"),
        "Model Name": cameraInfo.ModelName.decode("utf-8"),
        "Lens Description": cameraInfo.LensDescription.decode("utf-8")
    }

    return info

@PxLApi._eventProcessFunction
def camera_lost(hCamera, eventId, eventTimestamp, numDataBytes, data, userData):

    # Copy event specific data, if it is provided
    eventData = data.contents if bool(data) == True else 0

    print("EventCallbackFunction: hCamera=%s, eventId=%d" % (hex(hCamera), eventId))
    print("     eventTimestamp=%f, numDataBytes=%d" % (eventTimestamp, numDataBytes))
    print("     eventData=%s, userData=%d (%s)\n" % (hex(eventData), userData, hex(userData)))
    return PxLApi.ReturnCode.ApiSuccess