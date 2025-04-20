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
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.data import Axis, DataFromPlugins, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters
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

    library_initialized = False

    params = comon_parameters + [
        {'title': 'Camera Identifiers', 'name': 'ID', 'type': 'group', 'children': [
            {'title': 'Camera Index:', 'name': 'camera_index', 'type': 'list', 'value': 0, 'default': 0, 'limits': [0, 1]},
            {'title': 'Camera Model:', 'name': 'camera_model', 'type': 'str', 'value': '', 'readonly': True},
            {'title': 'Camera User ID:', 'name': 'camera_user_id', 'type': 'str', 'value': ''}
        ]},
        
        {'title': 'Resolution', 'name': 'resolution', 'type': 'list', 'value': [1280, 960], 'limits': [[320, 240], [640, 480], [1024, 768], [1280, 720], [1280, 960]]},
        {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1280, 'readonly': True},
        {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 960, 'readonly': True},
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

    callback_signal = pyqtSignal()

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: ic4.Grabber = None
        self.device_info = None
        self.user_id = "DMK Camera"
        self.device_idx = 0
        self.map = None
        self.gui_data = None
        self.listener = None
        self.sink: ic4.QueueSink = None
        self.callback_thread = None

        self._crosshair = None
        self.x_axis = None
        self.y_axis = None
        self.axes = None
        self.data_shape = None


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
            self.ini_detector(controller=self.controller)
        elif param.name() == "camera_user_id":
            try:
                if self.device_info.model_name == 'DMK 33GR0134':
                    self.controller.device_property_map.set_value('DeviceUserID', param.value())
                    self.user_id = param.value()
                elif self.device_info.model_name == 'DMK 42BUC03':
                    self.user_id = param.value()
            except ic4.IC4Exception:
                pass
        elif param.name() == "resolution":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.WIDTH, param.value()[0])
                self.controller.device_property_map.set_value(ic4.PropId.HEIGHT, param.value()[1])
                if self.controller != None:
                    self.close()
                self.ini_detector(controller=self.controller)
                self._prepare_view()
            except ic4.IC4Exception:
                pass
        elif param.name() == "brightness":
            try:
                self.controller.device_property_map.set_value('Brightness', param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "contrast":
            try:
                self.controller.device_property_map.set_value('Contrast', param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "exposure_auto":
            try:
                if self.device_info.model_name == 'DMK 42BUC03':
                    self.controller.device_property_map.set_value('Exposure_Auto', param.value())
                elif self.device_info.model_name == 'DMK 33GR0134':
                    self.controller.device_property_map.set_value('ExposureAuto', param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "exposure_time":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.EXPOSURE_TIME, param.value() * 1e3)
            except ic4.IC4Exception:
                pass
        elif param.name() == "gain_auto":
            try:
                if self.device_info.model_name == 'DMK 42BUC03':
                    self.controller.device_property_map.set_value('Gain_Auto', param.value())
                elif self.device_info.model_name == 'DMK 33GR0134':
                    self.controller.device_property_map.set_value('GainAuto', param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "gain_value":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.GAIN, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "frame_rate":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.ACQUISITION_FRAME_RATE, param.value())
            except ic4.IC4Exception:
                pass
        elif param.name() == "gamma":
            try:
                self.controller.device_property_map.set_value(ic4.PropId.GAMMA, param.value())
            except ic4.IC4Exception:
                pass


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
        if not DAQ_2DViewer_DMK.library_initialized:
            ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)
            DAQ_2DViewer_DMK.library_initialized = True


        self.ini_detector_init(old_controller=controller,
                               new_controller=ic4.Grabber())
    
        devices = ic4.DeviceEnum.devices()
        self.device_idx = self.settings.child('ID','camera_index').value()
        
        # Get the device info of chosen camera index and open the device
        # If opening fails, try next device after small sleep to avoid library errors
        if self.device_idx == 0:
            self.device_info = devices[0]
            self.controller.device_open(self.device_info)
        else:
            while self.device_idx < len(devices):
                try:
                    self.device_info = devices[self.device_idx]
                    self.controller.device_open(self.device_info)
                    break
                except ic4.IC4Exception:
                    self.device_idx += 1
                    time.sleep(1.5)
            else:
                raise RuntimeError("No available devices could be opened.")

        # Get device properties and set pixel format to Mono8 (Mono16) depending on the camera model
        self.map = self.controller.device_property_map
        if self.device_info.model_name == 'DMK 42BUC03':
            self.controller.device_property_map.try_set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.Mono8)
        elif self.device_info.model_name == 'DMK 33GR0134':
            self.controller.device_property_map.try_set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.Mono16)

        # Set param values for configuration based on camera in use (maybe compactify this later, but it's working..)
        self.settings.child('ID','camera_model').setValue(self.device_info.model_name)

        if self.device_info.model_name == 'DMK 33GR0134':
            self.settings.child('ID','camera_user_id').setValue(self.map.get_value_str('DeviceUserID'))
        elif self.device_info.model_name == 'DMK 42BUC03':
            self.settings.child('ID','camera_user_id').setValue(self.user_id)
        try:
            self.settings.param('brightness').setValue(self.map.get_value_float('Brightness'))
            self.settings.param('brightness').setDefault(self.map.get_value_float('Brightness'))
            self.settings.param('brightness').setLimits([self.map['Brightness'].minimum, self.map['Brightness'].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('contrast').setValue(self.map.get_value_float('Contrast'))
            self.settings.param('contrast').setDefault(self.map.get_value_float('Contrast'))
            self.settings.param('contrast').setLimits([self.map['Contrast'].minimum, self.map['Contrast'].maximum])
        except ic4.IC4Exception:
            pass
        try:
            if self.device_info.model_name == 'DMK 42BUC03':
                self.settings.child('exposure', 'exposure_auto').setValue(self.map.get_value_bool('Exposure_Auto'))
            elif self.device_info.model_name == 'DMK 33GR0134':
                self.settings.child('exposure', 'exposure_auto').setValue(self.map.get_value_bool('ExposureAuto'))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('exposure', 'exposure_time').setValue(self.map.get_value_float(ic4.PropId.EXPOSURE_TIME) * 1e-3)
            self.settings.child('exposure', 'exposure_time').setDefault(self.map.get_value_float(ic4.PropId.EXPOSURE_TIME) * 1e-3)
            self.settings.child('exposure', 'exposure_time').setLimits([self.map[ic4.PropId.EXPOSURE_TIME].minimum  * 1e-3, self.map[ic4.PropId.EXPOSURE_TIME].maximum  * 1e-3])
        except ic4.IC4Exception:
            pass
        try:
            if self.device_info.model_name == 'DMK 42BUC03':
                self.settings.child('gain', 'gain_auto').setValue(self.map.get_value_bool('Gain_Auto'))
            elif self.device_info.model_name == 'DMK 33GR0134':
                self.settings.child('gain', 'gain_auto').setValue(self.map.get_value_bool('GainAuto'))
        except ic4.IC4Exception:
            pass
        try:
            self.settings.child('gain', 'gain_value').setValue(self.map.get_value_float(ic4.PropId.GAIN))
            self.settings.child('gain', 'gain_value').setDefault(self.map.get_value_float(ic4.PropId.GAIN))
            self.settings.child('gain', 'gain_value').setLimits([self.map[ic4.PropId.GAIN].minimum, self.map[ic4.PropId.GAIN].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('frame_rate').setValue(self.map.get_value_float(ic4.PropId.ACQUISITION_FRAME_RATE))
            self.settings.param('frame_rate').setDefault(self.map.get_value_float(ic4.PropId.ACQUISITION_FRAME_RATE))
            self.settings.param('frame_rate').setLimits([self.map[ic4.PropId.ACQUISITION_FRAME_RATE].minimum, self.map[ic4.PropId.ACQUISITION_FRAME_RATE].maximum])
        except ic4.IC4Exception:
            pass
        try:
            self.settings.param('gamma').setValue(self.map.get_value_float(ic4.PropId.GAMMA))
            self.settings.param('gamma').setDefault(self.map.get_value_float(ic4.PropId.GAMMA))
            self.settings.param('gamma').setLimits([self.map[ic4.PropId.GAMMA].minimum, self.map[ic4.PropId.GAMMA].maximum])
        except ic4.IC4Exception:
            pass

        # Stream setup for data acquisition
        self.gui_data = {"ready": False, 
                         "frame": np.zeros((self.controller.device_property_map.get_value_int(ic4.PropId.WIDTH), 
                                           self.controller.device_property_map.get_value_int(ic4.PropId.HEIGHT)))}

        # Create the listener callback object
        listener_callback = ListenerCallback(self.gui_data)
        listener_callback.data_ready.connect(self.emit_data)  # When data is ready, call emit_data

        # Create a Qt thread to run the listener callback
        self.callback_thread = QtCore.QThread()
        listener_callback.moveToThread(self.callback_thread)

        # Connect the signal to trigger the acquisition process
        self.callback_signal.connect(listener_callback.wait_for_acquisition)

        # Start the thread
        self.callback_thread.start()

        # Now connect QueueSink to the listener
        self.sink = ic4.QueueSink(listener_callback.get_listener(), max_output_buffers=1)
        listener_callback.get_listener().sink = self.sink  # Store the sink in the listener

        # Initialize the stream
        self.controller.stream_setup(self.sink)

        self._prepare_view()
        info = "Imaging Source camera initialized"
        print(f"{self.device_info.model_name} camera initialized successfully")
        initialized = True
        return info, initialized
    
    def _prepare_view(self):
        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs are changed"""

        width = self.controller.device_property_map.get_value_int(ic4.PropId.WIDTH)
        height = self.controller.device_property_map.get_value_int(ic4.PropId.HEIGHT)

        self.settings.param('width').setValue(width)
        self.settings.param('height').setValue(height)

        mock_data = np.zeros((width, height))

        self.x_axis = Axis(data=np.linspace(1, width, width), label='Pixels', index = 1)

        # Switches viewer type depending on image size
        if height != 1:
            self.y_axis = Axis(data=np.linspace(1, height, height), label='Pixels', index = 1)

            if width != 1:
                data_shape = "Data2D"
                axes = [self.x_axis, self.y_axis]

            else:
                data_shape = "Data1D"
                axes = [self.y_axis]
        else:
            data_shape = "Data1D"
            self.x_axis.index = 0
            axes = [self.x_axis]

        if data_shape != self.data_shape:
            self.data_shape = data_shape
            self.axes = axes
            self.dte_signal_temp.emit(
                DataToExport(f'{self.user_id}',
                             data=[DataFromPlugins(name=f'{self.user_id}',
                                                   data=[np.squeeze(np.zeros((height, width)))],
                                                   dim=self.data_shape,
                                                   labels=[f'{self.user_id}_{self.data_shape}'],
                                                   axes=self.axes)]))
        
            QtWidgets.QApplication.processEvents()

    def close(self):
        """Terminate the communication protocol"""
        if self.controller.is_acquisition_active:
            self.controller.acquisition_stop()
        self.controller.device_close()

        if self.callback_thread is not None:
            self.callback_thread.quit()
            self.callback_thread.wait()
            self.callback_thread = None

        self.controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""           
        print(f"{self.device_info.model_name} communication terminated successfully")   

    def grab_data(self, Naverage=1, **kwargs):
        """
        Grabs the data. Synchronous method (kinda).
        ----------
        Naverage: (int) Number of averaging
        kwargs: (dict) of others optionals arguments
        """

        try:
            self._prepare_view()
            if not self.controller.is_acquisition_active:
                self.controller.acquisition_start()

            self.callback_signal.emit()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

            
    def emit_data(self):
        try:
            frame = self.gui_data["frame"]
            if frame is not None:
                width = self.settings.param('width').value()
                height = self.settings.param('height').value()
                self.dte_signal.emit(
                    DataToExport(f'{self.user_id}', data=[DataFromPlugins(
                        name=f'{self.user_id}',
                        data=[np.squeeze(frame.reshape(height, width))],
                        dim=self.data_shape,
                        labels=[f'{self.user_id}_{self.data_shape}'],
                        axes=self.axes)]))
                
                self.gui_data["ready"] = False
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))


    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        self.controller.acquisition_stop()
        return ''
    
    def crosshair(self, crosshair_info, ind_viewer=0):
        sleep_ms = 10
        ind = 0

        while not self.gui_data["ready"]:
            QtCore.QThread.msleep(sleep_ms)
            QtWidgets.QApplication.processEvents()

            ind += 1
            if ind * sleep_ms >= 1000:
                self.emit_status(ThreadCommand('_raise_timeout'))
                break

        

class ListenerCallback(QtCore.QObject):
    data_ready = pyqtSignal()  # Signal when data is ready

    def __init__(self, gui_data):
        super().__init__()
        self.gui_data = gui_data
        self.listener = Listener(gui_data=self.gui_data, signal_emitter=self)

    def get_listener(self):
        return self.listener

    def wait_for_acquisition(self):
        # You need to manually trigger frames_queued and pass the sink
        if self.listener.sink:
            self.listener.frames_queued(self.listener.sink)
        
        new_data = self.gui_data["frame"]
        if new_data is not None:  # Check if new data is available
            self.data_ready.emit()  # Emit signal when data is ready

class Listener(ic4.QueueSinkListener):
    def __init__(self, gui_data, signal_emitter):
        super().__init__()
        self.gui_data = gui_data
        self.signal_emitter = signal_emitter
        self.sink = None  # This will be populated later by QueueSink

    def frames_queued(self, sink: ic4.QueueSink):
        # Logic to handle frames once they are queued
        buffer = sink.try_pop_output_buffer()
        if buffer is not None:
            self.gui_data["frame"] = buffer.numpy_copy()
            self.gui_data["ready"] = True
            buffer.release()

    def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
        self.sink = sink  # Save the sink reference when connected
        return True

    def sink_disconnected(self, sink: ic4.QueueSink):
        # Handle when sink is disconnected
        self.sink = None

def main():
    """
    this method start a DAQ_Viewer object with this defined plugin as detector
    Returns
    -------

    """
    import sys
    from PyQt6 import QtWidgets
    from pymodaq.utils.gui_utils import DockArea
    from pymodaq.control_modules.daq_viewer import DAQ_Viewer
    from pathlib import Path

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle("PyMoDAQ Viewer")
    detector = Path(__file__).stem[13:]
    det_type = f"DAQ{Path(__file__).stem[4:6].upper()}"
    prog = DAQ_Viewer(area, title="Testing")
    win.show()
    prog.daq_type = det_type
    prog.detector = detector
    prog.init_hardware_ui()

    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        main()
    finally:
        ic4.Library.exit()