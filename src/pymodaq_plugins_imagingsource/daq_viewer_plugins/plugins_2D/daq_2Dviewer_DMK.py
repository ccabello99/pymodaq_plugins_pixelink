import numpy as np
import time
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from PyQt5.QtCore import pyqtSignal
import imagingcontrol4 as ic4

from qtpy import QtWidgets, QtCore

# TODO:
# (1) change the name of the following class to DAQ_2DViewer_TheNameOfYourChoice
# (2) change the name of this file to daq_2Dviewer_TheNameOfYourChoice ("TheNameOfYourChoice" should be the SAME
#     for the class name and the file name.)
# (3) this file should then be put into the right folder, namely IN THE FOLDER OF THE PLUGIN YOU ARE DEVELOPING:
#     pymodaq_plugins_my_plugin/daq_viewer_plugins/plugins_2D
class DAQ_2DViewer_DMK(DAQ_Viewer_base):
    """ Instrument plugin class for a 2D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    
    * Tested with DMK 42BUC03 camera.
    * PyMoDAQ version 5.0.2
    * Tested on Windows 11
    * Installation instructions: For this camera, you need to install the Imaging Source drivers, 
                                 specifically "Device Driver for USB Cameras" in legacy software
                                 and the Python wrapper

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.

    """

    library_initialized = False

    params = comon_parameters + [
        {'title': 'Camera Index:', 'name': 'camera_index', 'type': 'list', 'value': 0, 'default': 0, 'limits': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {'title': 'Camera Model:', 'name': 'camera_model', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1280, 'default': 1280, 'limits': [96, 1280]},
        {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 960, 'default': 960, 'limits': [96, 960]},
        {'title': 'Exposure', 'name': 'exposure', 'type': 'group', 'children': [
            {'title': 'Auto Exposure', 'name': 'exposure_auto', 'type': 'list', 'value': "Off", 'default': "Off", 'limits': ['On', 'Off']},
            {'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'float', 'value': 100.0, 'default': 100.0, 'limits': [100.0, 30000000.0]}
        ]},
        {'title': 'Gain', 'name': 'gain', 'type': 'group', 'children': [
            {'title': 'Auto Gain', 'name': 'gain_auto', 'type': 'list', 'value': "Off", 'default': "Off", 'limits': ['On', 'Off']},
            {'title': 'Value', 'name': 'gain_value', 'type': 'float', 'value': 34.0, 'default': 34.0, 'limits': [34.0, 255.0]}
        ]},
        {'title': 'Frame Rate', 'name': 'frame_rate', 'type': 'float', 'value': 25.0, 'default': 25.0, 'limits': [7.5, 25.0]},
        {'title': 'Gamma', 'name': 'gamma', 'type': 'float', 'value': 1.0, 'default': 1.0, 'limits': [1.0, 500.0]}
        
    ]

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: ic4.Grabber = None
        self.device_info = None
        self.map = None
        self.gui_data = None
        self.listener = None
        self.sink: ic4.QueueSink = None

        self.data_shape = 'Data2D'

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "camera_index":
            if self.controller != None:
                self.close()
            time.sleep(2)
            self.ini_detector(controller=self.controller, device_idx=param.value())
        elif param.name() == "width":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.WIDTH, param.value())
                print(self.controller.device_property_map.get_value_int(ic4.PropId.WIDTH))
            except ic4.IC4Exception:
                pass
        elif param.name() == "height":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.HEIGHT, param.value())
                print(self.controller.device_property_map.get_value_int(ic4.PropId.HEIGHT))
            except ic4.IC4Exception:
                pass
        elif param.name() == "exposure_auto":
            try:
                self.controller.device_property_map.set_value('Exposure_Auto', param.value())
                print(self.controller.device_property_map.get_value_bool('Exposure_Auto'))
            except ic4.IC4Exception:
                pass
            try:
                self.controller.device_property_map.set_value('ExposureAuto', param.value())
                print(self.controller.device_property_map.get_value_bool('ExposureAuto'))
            except ic4.IC4Exception:
                pass
        elif param.name() == "exposure_time":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.EXPOSURE_TIME, param.value())
                print(self.controller.device_property_map.get_value_float(ic4.PropId.EXPOSURE_TIME))
            except ic4.IC4Exception:
                pass
        elif param.name() == "gain_auto":
            try:
                self.controller.device_property_map.set_value('Gain_Auto', param.value())
                print(self.controller.device_property_map.get_value_bool('Gain_Auto'))
            except ic4.IC4Exception:
                pass
            try:
                self.controller.device_property_map.set_value('GainAuto', param.value())
                print(self.controller.device_property_map.get_value_bool('GainAuto'))
            except ic4.IC4Exception:
                pass
        elif param.name() == "gain_value":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.GAIN, param.value())
                print(self.controller.device_property_map.get_value_float(ic4.PropId.GAIN))
            except ic4.IC4Exception:
                pass
        elif param.name() == "frame_rate":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.ACQUISITION_FRAME_RATE, param.value())
                print(self.controller.device_property_map.get_value_float(ic4.PropId.ACQUISITION_FRAME_RATE))
            except ic4.IC4Exception:
                pass
        elif param.name() == "gamma":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.GAMMA, param.value())
                print(self.controller.device_property_map.get_value_float(ic4.PropId.GAMMA))
            except ic4.IC4Exception:
                pass


    def ini_detector(self, controller=None, device_idx=0):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """

        if not DAQ_2DViewer_DMK.library_initialized:
            ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)
            DAQ_2DViewer_DMK.library_initialized = True


        self.ini_detector_init(old_controller=controller,
                               new_controller=ic4.Grabber())
        
        # Get number of available cameras and list them in the camera_index parameter
        self.settings.param('camera_index').setLimits([0, len(ic4.DeviceEnum.devices())-1])
        
        # Get the device info of chosen camera index and open the device
        self.device_info = ic4.DeviceEnum.devices()[device_idx]
        self.controller.device_open(self.device_info)

        # Get device properties and set pixel format to Mono8
        self.map = self.controller.device_property_map
        self.controller.device_property_map.try_set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.Mono8)

        # Set param values for configuration based on camera in use
        self.settings.param('camera_model').setValue(self.device_info.model_name)

        try:
            self.settings.param('width').setValue(self.map.get_value_int(ic4.PropId.WIDTH))
            self.settings.param('width').setDefault(self.map.get_value_int(ic4.PropId.WIDTH))
            self.settings.param('width').setLimits([self.map[ic4.PropId.WIDTH].minimum, self.map[ic4.PropId.WIDTH].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('height').setValue(self.map.get_value_int(ic4.PropId.HEIGHT))
            self.settings.param('height').setDefault(self.map.get_value_int(ic4.PropId.HEIGHT))
            self.settings.param('width').setLimits([self.map[ic4.PropId.HEIGHT].minimum, self.map[ic4.PropId.HEIGHT].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('exposure', 'exposure_auto').setValue(self.map.get_value_bool('Exposure_Auto'))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('exposure', 'exposure_auto').setValue(self.map.get_value_bool('ExposureAuto'))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('exposure', 'exposure_time').setValue(self.map.get_value_float(ic4.PropId.EXPOSURE_TIME))
            self.settings.child('exposure', 'exposure_time').setDefault(self.map.get_value_float(ic4.PropId.EXPOSURE_TIME))
            self.settings.param('width').setLimits([self.map[ic4.PropId.EXPOSURE_TIME].minimum, self.map[ic4.PropId.EXPOSURE_TIME].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('gain', 'gain_auto').setValue(self.map.get_value_bool('Gain_Auto'))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('gain', 'gain_auto').setValue(self.map.get_value_bool('GainAuto'))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('gain', 'gain_value').setValue(self.map.get_value_float(ic4.PropId.GAIN))
            self.settings.child('gain', 'gain_value').setDefault(self.map.get_value_float(ic4.PropId.GAIN))
            self.settings.param('width').setLimits([self.map[ic4.PropId.GAIN].minimum, self.map[ic4.PropId.GAIN].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('frame_rate').setValue(self.map.get_value_float(ic4.PropId.ACQUISITION_FRAME_RATE))
            self.settings.param('frame_rate').setDefault(self.map.get_value_float(ic4.PropId.ACQUISITION_FRAME_RATE))
            self.settings.param('width').setLimits([self.map[ic4.PropId.ACQUISITION_FRAME_RATE].minimum, self.map[ic4.PropId.ACQUISITION_FRAME_RATE].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('gamma').setValue(self.map.get_value_float(ic4.PropId.GAMMA))
            self.settings.param('gamma').setDefault(self.map.get_value_float(ic4.PropId.GAMMA))
            self.settings.param('width').setLimits([self.map[ic4.PropId.GAMMA].minimum, self.map[ic4.PropId.GAMMA].maximum])
        except ic4.IC4Exception:
            pass
        
        # Stream setup for data acquisition
        self.gui_data = {"ready": False, "image": np.zeros(1)}
        self.listener = Listener(self.gui_data)
        self.sink = ic4.QueueSink(self.listener, max_output_buffers=1)
        self.controller.stream_setup(self.sink)
        self.sink.alloc_and_queue_buffers(10)


        info = "Imaging Source camera initialized"
        print(f"{self.device_info.model_name} camera initialized successfully")
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        if self.controller.is_streaming:
            self.controller.stream_stop()

        self.controller.device_close()
        self.controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""
            
        print(f"Camera communication terminated successfully")   

    def grab_data(self, Naverage=1, **kwargs):
        """
        Grabs the data. Synchronous method (kinda).
        ----------
        Naverage: (int) Number of averaging
        kwargs: (dict) of others optionals arguments
        """

        # Set sleep time to match frame rate to avoid overloading the camera (cleaner solution in future?)
        time.sleep(1/self.controller.device_property_map.get_value_float(ic4.PropId.ACQUISITION_FRAME_RATE))
        self.gui_data["ready"] = False
        self.emit_data()
            
    def emit_data(self):
        """
            Fonction used to emit data obtained by callback.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        try:
            image = self.gui_data["image"]
            if image is not None:
                self.data_grabed_signal.emit([DataFromPlugins(name='DMK Camera',
                                                              data=[np.squeeze(image)],
                                                              dim=self.data_shape,
                                                              labels=[f'DMK_{self.data_shape}'])])



            # To make sure that timed events are executed in continuous grab mode
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))


    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        return ''


class Listener(ic4.QueueSinkListener):
    def __init__(self, gui_data):
        self.gui_data = gui_data

    def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
        return True
    def sink_disconnected(self, sink: ic4.QueueSink):
        pass

    def frames_queued(self, sink: ic4.QueueSink):
        buffer = sink.try_pop_output_buffer()
        self.gui_data["image"] = buffer.numpy_copy()
        self.gui_data["ready"] = True
        buffer.release()
    
class ImagingSourceCallback(QtCore.QObject):
    """Callback object """
    data_sig = pyqtSignal()

    def __init__(self, wait_fn):
        super().__init__()
        # Set the wait function
        self.wait_fn = wait_fn

    def wait_for_acquisition(self):
        new_data = self.wait_fn()
        if new_data is not False:  # will be returned if the main thread called CancelWait
            self.data_sig.emit()


if __name__ == '__main__':
    try:
        main(__file__)
    finally:
        ic4.Library.exit()