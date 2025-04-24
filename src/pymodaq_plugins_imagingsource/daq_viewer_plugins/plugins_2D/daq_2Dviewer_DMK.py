import numpy as np
import time
import imagingcontrol4 as ic4

import warnings
import numpy as np
# Suppress only NumPy RuntimeWarnings (bc of crosshair bug)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


from pymodaq.utils.daq_utils import (
    ThreadCommand,
    getLineInfo,
)
from pymodaq_plugins_imagingsource.hardware.imagingsource import ImagingSourceCamera, Listener
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.data import Axis, DataFromPlugins, DataToExport
from pymodaq.control_modules.viewer_utility_classes import main, DAQ_Viewer_base, comon_parameters
from PyQt6.QtCore import pyqtSignal
from qtpy import QtWidgets, QtCore


class DAQ_2DViewer_DMK(DAQ_Viewer_base):
    """ Instrument plugin class for a 2D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    
    * Tested with DMK 42BUC03/33GR0134 camera.
    * PyMoDAQ version 5.0.2
    * Tested on Windows 11
    * Installation instructions: For this camera, you need to install the Imaging Source drivers, 
                                 specifically "Device Driver for USB Cameras" and/or "Device Driver for GigE Cameras" in legacy software

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.

    """

    try:
        ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)
    except RuntimeError:
        pass # library already initialized

    live_mode_available = True

    devices = ic4.DeviceEnum.devices()
    camera_list = [device.model_name for device in devices]

    params = comon_parameters + [
        {'title': 'Camera List:', 'name': 'camera_list', 'type': 'list', 'value': '', 'limits': camera_list},
        {'title': 'Camera Identifiers', 'name': 'ID', 'type': 'group', 'children': [
            {'title': 'Camera Model:', 'name': 'camera_model', 'type': 'str', 'value': '', 'readonly': True},
            {'title': 'Camera Serial Number:', 'name': 'camera_serial', 'type': 'str', 'value': '', 'readonly': True},
            {'title': 'Camera User ID:', 'name': 'camera_user_id', 'type': 'str', 'value': ''}
        ]},  
        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
            {'title': 'Update ROI', 'name': 'update_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Binning', 'name': 'binning', 'type': 'list', 'limits': [1, 2], 'default': 1},
            {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1280, 'readonly': True},
            {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 960, 'readonly': True},
        ]},
        {'title': 'Brightness', 'name': 'brightness', 'type': 'slide', 'value': 1.0, 'default': 1.0, 'limits': [0.0, 1.0]},
        {'title': 'Contrast', 'name': 'contrast', 'type': 'slide', 'value': 1.0, 'default': 1.0, 'limits': [0.0, 1.0]},
        {'title': 'Exposure', 'name': 'exposure', 'type': 'group', 'children': [
            {'title': 'Auto Exposure', 'name': 'exposure_auto', 'type': 'led_push', 'value': "Off", 'default': "Off", 'limits': ['On', 'Off']},
            {'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'float', 'value': 100.0, 'default': 100.0, 'limits': [0.0, 1.0]}
        ]},
        {'title': 'Gain', 'name': 'gain', 'type': 'group', 'children': [
            {'title': 'Auto Gain', 'name': 'gain_auto', 'type': 'led_push'},
            {'title': 'Value', 'name': 'gain_value', 'type': 'slide', 'value': 1.0, 'default': 1.0, 'limits': [0.0, 1.0]}
        ]},
        {'title': 'Frame Rate', 'name': 'frame_rate', 'type': 'slide', 'value': 1.0, 'default': 1.0, 'limits': [0.0, 1.0]},
        {'title': 'Gamma', 'name': 'gamma', 'type': 'slide', 'value': 1.0, 'default': 1.0, 'limits': [0.0, 1.0]}        
    ]

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: None
        self.user_id = None

        self.x_axis = None
        self.y_axis = None
        self.axes = None
        self.data_shape = None

    def init_controller(self) -> ImagingSourceCamera:
        # Define the camera controller.
        # Use any argument necessary (serial_number, camera index, etc.) depending on the camera

        # Init camera with currently selected user id name
        list_param = self.settings.param('camera_list')
        self.user_id = list_param.value()
        self.emit_status(ThreadCommand('Update_Status', [f"Trying to connect to {self.user_id}", 'log']))
        devices = ic4.DeviceEnum.devices()
        camera_list = self.get_camera_list()
        for cam in camera_list:
            if cam == self.user_id:
                device_idx = camera_list.index(self.user_id)
                device_info = devices[device_idx]
                return ImagingSourceCamera(info=device_info, callback=self.emit_data_callback)
        self.emit_status(ThreadCommand('Update_Status', ["Camera not found", 'log']))
        raise ValueError(f"Camera with name {self.user_id} not found anymore.")

    def ini_detector(self, controller=None):
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

        # Initialize the Imaging Source library if not already done
        # This is done to avoid multiple initializations of the library
        # which can cause issues with the camera operation
        #if not DAQ_2DViewer_DMK.library_initialized:
        #    ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)
        #    DAQ_2DViewer_DMK.library_initialized = True


        self.ini_detector_init(old_controller=controller,
                               new_controller=self.init_controller())

        # Get device properties and set pixel format to Mono8 (Mono16) depending on the camera model
        map = self.controller.camera.device_property_map
        if self.controller.model_name == 'DMK 42BUC03':
            self.controller.camera.device_property_map.try_set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.Mono8)
        elif self.controller.model_name == 'DMK 33GR0134':
            self.controller.camera.device_property_map.try_set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.Mono16)

        # Set param values for configuration based on camera in use (maybe compactify this later, but it's working..)
        self.settings.child('ID','camera_model').setValue(self.controller.model_name)
        self.settings.child('ID','camera_serial').setValue(self.controller.device_info.serial)
        self.settings.child('ID', 'camera_user_id').setValue(self.user_id)

        # Initialize the stream but defer acquisition start until we start grabbing
        self.controller.setup_acquisition()

        if self.controller.model_name == 'DMK 33GR0134':
            self.settings.child('ID','camera_user_id').setValue(map.get_value_str('DeviceUserID'))
        elif self.controller.model_name == 'DMK 42BUC03':
            self.settings.child('ID','camera_user_id').setValue(self.user_id)
        try:
            self.settings.param('brightness').setValue(map.get_value_float(self.controller.brightness))
            self.settings.param('brightness').setDefault(map.get_value_float(self.controller.brightness))
            self.settings.param('brightness').setLimits([map[self.controller.brightness].minimum, map[self.controller.brightness].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('contrast').setValue(map.get_value_float(self.controller.contrast))
            self.settings.param('contrast').setDefault(map.get_value_float(self.controller.contrast))
            self.settings.param('contrast').setLimits([map[self.controller.contrast].minimum, map[self.controller.contrast].maximum])
        except ic4.IC4Exception:
            pass
        try:
            if self.controller.model_name == 'DMK 42BUC03':
                self.settings.child('exposure', 'exposure_auto').setValue(map.get_value_bool(self.controller.exposure_auto))
            elif self.controller.model_name == 'DMK 33GR0134':
                self.settings.child('exposure', 'exposure_auto').setValue(map.get_value_bool(self.controller.exposure_auto))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('exposure', 'exposure_time').setValue(map.get_value_float(self.controller.exposure_time) * 1e-3)
            self.settings.child('exposure', 'exposure_time').setDefault(map.get_value_float(self.controller.exposure_time) * 1e-3)
            self.settings.child('exposure', 'exposure_time').setLimits([map[self.controller.exposure_time].minimum  * 1e-3, map[self.controller.exposure_time].maximum  * 1e-3])
        except ic4.IC4Exception:
            pass
        try:
            if self.controller.model_name == 'DMK 42BUC03':
                self.settings.child('gain', 'gain_auto').setValue(map.get_value_bool(self.controller.gain_auto))
            elif self.controller.model_name == 'DMK 33GR0134':
                self.settings.child('gain', 'gain_auto').setValue(map.get_value_bool(self.controller.gain_auto))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('gain', 'gain_value').setValue(map.get_value_float(self.controller.gain_value))
            self.settings.child('gain', 'gain_value').setDefault(map.get_value_float(self.controller.gain_value))
            self.settings.child('gain', 'gain_value').setLimits([map[self.controller.gain_value].minimum, map[self.controller.gain_value].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('frame_rate').setValue(map.get_value_float(self.controller.frame_rate))
            self.settings.param('frame_rate').setDefault(map.get_value_float(self.controller.frame_rate))
            self.settings.param('frame_rate').setLimits([map[self.controller.frame_rate].minimum, map[self.controller.frame_rate].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('gamma').setValue(map.get_value_float(self.controller.gamma))
            self.settings.param('gamma').setDefault(map.get_value_float(self.controller.gamma))
            self.settings.param('gamma').setLimits([map[self.controller.gamma].minimum, map[self.controller.gamma].maximum])
        except ic4.IC4Exception:
            pass

        self._prepare_view()
        info = "Initialized camera"
        print(f"{self.user_id} camera initialized successfully")
        initialized = True
        return info, initialized
    
    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "camera_list":
            if self.controller != None:
                self.close()
            self.ini_detector()
        elif param.name() == "camera_user_id":
            try:
                if self.controller.model_name == 'DMK 33GR0134':
                    self.controller.camera.device_property_map.set_value('DeviceUserID', param.value())
                    self.user_id = param.value()
                elif self.controller.model_name == 'DMK 42BUC03':
                    self.user_id = param.value()
            except ic4.IC4Exception:
                pass
        elif param.name() == "update_roi":
            if param.value():  # Switching on ROI

                # We handle ROI and binning separately for clarity
                (old_x, _, old_y, _, xbin, ybin) = self.controller.get_roi()  # Get current binning

                y0, x0 = self.roi_info.origin.coordinates
                height, width = self.roi_info.size.coordinates

                # Values need to be rescaled by binning factor and shifted by current x0,y0 to be correct.
                new_x = (old_x + x0) * xbin
                new_y = (old_y + y0) * xbin
                new_width = width * ybin
                new_height = height * ybin

                new_roi = (new_x, new_width, xbin, new_y, new_height, ybin)
                self.update_rois(new_roi)
                param.setValue(False)
        elif param.name() == 'binning':
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings.child('roi', 'binning').value()
            ybin = self.settings.child('roi', 'binning').value()
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)
        elif param.name() == "clear_roi":
            if param.value():  # Switching on ROI
                wdet, hdet = self.controller.get_detector_size()
                # self.settings.child('ROIselect', 'x0').setValue(0)
                # self.settings.child('ROIselect', 'width').setValue(wdet)
                self.settings.child('roi', 'binning').setValue(1)
                #
                # self.settings.child('ROIselect', 'y0').setValue(0)
                # new_height = self.settings.child('ROIselect', 'height').setValue(hdet)

                new_roi = (0, wdet, 1, 0, hdet, 1)
                self.update_rois(new_roi)
                param.setValue(False)
        elif param.name() == "brightness":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.brightness, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "contrast":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.brightness, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "exposure_auto":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.exposure_auto, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "exposure_time":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.exposure_time, param.value() * 1e3)
            except ic4.IC4Exception:
                pass
        elif param.name() == "gain_auto":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.gain_auto, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "gain_value":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.gain_value, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "frame_rate":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.frame_rate, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "gamma":
            try:
                self.controller.camera.device_property_map.set_value(self.controller.gamma, param.value())
            except ic4.IC4Exception:
                pass
    
    def _prepare_view(self):
         """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
         ROIs are changed"""
 
         width = self.controller.camera.device_property_map.get_value_int(ic4.PropId.WIDTH)
         height = self.controller.camera.device_property_map.get_value_int(ic4.PropId.HEIGHT)
 
         self.settings.child('roi', 'width').setValue(width)
         self.settings.child('roi', 'height').setValue(height)
 
         mock_data = np.zeros((width, height))

         self.x_axis = Axis(label='Pixels', data=np.linspace(1, width, width), index=0)
 
         if width != 1 and height != 1:
             data_shape = 'Data2D'
             self.y_axis = Axis(label='Pixels', data=np.linspace(1, height, height), index=1)
             self.axes = [self.x_axis, self.y_axis]
         else:
             data_shape = 'Data1D'
             self.axes = [self.x_axis]
 
         if data_shape != self.data_shape:
             self.data_shape = data_shape
             self.dte_signal_temp.emit(
                 DataToExport(f'{self.user_id}',
                              data=[DataFromPlugins(name=f'{self.user_id}',
                                                    data=[np.squeeze(mock_data)],
                                                    dim=self.data_shape,
                                                    labels=[f'{self.user_id}_{self.data_shape}'],
                                                    axes=self.axes)]))
 
             QtWidgets.QApplication.processEvents()

    def update_rois(self, new_roi):
        (new_x, new_width, new_xbinning, new_y, new_height, new_ybinning) = new_roi
        if new_roi != self.controller.get_roi():
            # self.controller.set_attribute_value("ROIs",[new_roi])
            self.controller.set_roi(hstart=new_x,
                                    hend=new_x + new_width,
                                    vstart=new_y,
                                    vend=new_y + new_height,
                                    hbin=new_xbinning,
                                    vbin=new_ybinning)
            self.emit_status(ThreadCommand('Update_Status', [f'Changed ROI: {new_roi}']))
            self.close()
            self.ini_detector()
            # Finally, prepare view for displaying the new data
            self._prepare_view()

    def grab_data(self, Naverage: int = 1, live: bool = False, **kwargs) -> None:
        try:
            if live:
                self._prepare_view()
                self.controller.start_grabbing(frame_rate=self.settings.param('frame_rate').value())
            else:
                self._prepare_view()
                if not self.controller.camera.is_acquisition_active:
                    self.controller.camera.acquisition_start()
                QtCore.QTimer.singleShot(50, self.emit_data)
                if self.controller.camera.is_acquisition_active:
                    self.controller.camera.acquisition_stop()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

            
    def emit_data(self):
        """
            Function used to emit data obtained by callback.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        try:
            # Get data from buffer
            buffer = self.controller.sink.try_pop_output_buffer()
            if buffer is not None:
                frame = buffer.numpy_copy()
                buffer.release()
                # Emit the frame.
                self.dte_signal.emit(
                    DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                        name=f'{self.user_id}',
                        data=[np.squeeze(frame)],
                        dim=self.data_shape,
                        labels=[f'{self.user_id}_{self.data_shape}'],
                        axes=self.axes)]))

            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def emit_data_callback(self, frame) -> None:
        self.dte_signal.emit(
            DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                name=f'{self.user_id}',
                data=[np.squeeze(frame)],
                dim=self.data_shape,
                labels=[f'{self.user_id}_{self.data_shape}'],
                axes=self.axes)]))

    def stop(self):
        self.controller.camera.acquisition_stop()
        return ''
    
    def close(self):
        """Terminate the communication protocol"""
        self.controller.close()

        self.controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""           
        print(f"{self.user_id} communication terminated successfully")   
    
    def roi_select(self, roi_info, ind_viewer):
        self.roi_info = roi_info
    
    def crosshair(self, crosshair_info, ind_viewer=0):
        sleep_ms = 150
        QtCore.QTimer.singleShot(sleep_ms, QtWidgets.QApplication.processEvents)

    def get_camera_list(self):
        devices = ic4.DeviceEnum.devices()
        cameras = []
        for dev in devices:
            cameras.append(dev.model_name)
        return cameras


if __name__ == '__main__':
    try:
        main(__file__, init=False)
    finally:
        ic4.Library.exit()