import numpy as np
import time
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from PyQt5.QtCore import pyqtSignal
from pymodaq_plugins_imagingsource.hardware.ImagingSource import ic4, Listener, ImagingSourceCamera

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
    * PyMoDAQ version 5.0.1
    * Tested on Windows 11
    * Installation instructions: For this camera, you need to install the Imaging Source drivers, 
                                 specifically "Device Driver for USB Cameras" in legacy software
                                 and the Python wrapper

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.
         
    # TODO add your particular attributes here if any

    """
    params = params = comon_parameters + [
        {'title': 'Camera Model:', 'name': 'camera_name', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1280, 'default': 1280, 'min': 96, 'max': 1280},
        {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 960, 'default': 960, 'min': 96, 'max': 960},
        {'title': 'Exposure', 'name': 'exposure', 'type': 'group', 'children': [
            {'title': 'Auto Exposure', 'name': 'exposure_auto', 'type': 'list', 'value': "Off", 'limits': ['On', 'Off']},
            {'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'float', 'value': 100.0, 'default': 100.0, 'min': 100.0, 'max': 30000000.0}
        ]},
        {'title': 'Gain', 'name': 'gain', 'type': 'group', 'children': [
            {'title': 'Auto Gain', 'name': 'gain_auto', 'type': 'list', 'value': "Off", 'limits': ['On', 'Off']},
            {'title': 'Value', 'name': 'gain_value', 'type': 'float', 'value': 34, 'default': 34, 'min': 34, 'max': 255}
        ]},
        {'title': 'Frame Rate', 'name': 'frame_rate', 'type': 'float', 'value': 25, 'default': 25, 'min': 7.5, 'max': 25},
        {'title': 'Gamma', 'name': 'gamma', 'type': 'float', 'value': 1, 'default': 1, 'min': 1, 'max': 500}
        
    ]

    callback_signal = pyqtSignal()

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: ic4.Grabber = None
        self.device_info = None
        self.map = None
        self.listener = None
        self.sink = None

        self.x_axis = None
        self.y_axis = None

        self.data_shape = 'Data2D'
        self.callback_thread = None
        self.last_read_frame = None
        self.last_wait_frame = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "width":
            self.controller.device_property_map.set_value(ic4.PropId.WIDTH, param.value)
        elif param.name() == "height":
            self.controller.device_property_map.set_value(ic4.PropId.HEIGHT, param.value)
        elif param.name() == "auto_exposure":
            self.controller.device_property_map.set_value('Exposure_Auto', param.value)
        elif param.name() == "exposure_time":
            self.controller.device_property_map.set_value(ic4.PropId.EXPOSURE_TIME, param.value)
        elif param.name() == "auto_gain":
            self.controller.device_property_map.set_value('Gain_Auto', param.value)
        elif param.name() == "gain":
            self.controller.device_property_map.set_value(ic4.PropId.GAIN, param.value)
        elif param.name() == "frame_rate":
            self.controller.device_property_map.set_value(ic4.PropId.ACQUISITION_FRAME_RATE, param.value)
        elif param.name() == "gamma":
            self.controller.device_property_map.set_value(ic4.PropId.GAMMA, param.value)


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
        ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)


        self.ini_detector_init(old_controller=controller,
                               new_controller=ic4.Grabber())
        
        self.device_info = ic4.DeviceEnum.devices()[0]
        self.controller.device_open(self.device_info)
        self.settings.param('camera_name').setValue(self.device_info.model_name)
        self.map = self.controller.device_property_map
        self.controller.device_property_map.try_set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.Mono8)

        self.settings.param('width').setValue(self.map.get_value_int(ic4.PropId.WIDTH))
        self.settings.param('height').setValue(self.map.get_value_int(ic4.PropId.HEIGHT))
        self.settings.child('exposure', 'exposure_auto').setValue(self.map.get_value_bool('Exposure_Auto'))
        self.settings.child('exposure', 'exposure_time').setValue(self.map.get_value_float(ic4.PropId.EXPOSURE_TIME))
        self.settings.child('gain', 'gain_auto').setValue(self.map.get_value_bool('Gain_Auto'))
        self.settings.child('gain', 'gain_value').setValue(self.map.get_value_float(ic4.PropId.GAIN))
        self.settings.param('frame_rate').setValue(self.map.get_value_float(ic4.PropId.ACQUISITION_FRAME_RATE))
        self.settings.param('gamma').setValue(self.map.get_value_float(ic4.PropId.GAMMA))

        self.x_axis = Axis(data=np.linspace(1, self.map[ic4.PropId.WIDTH].maximum, self.map.get_value_int(ic4.PropId.WIDTH)))
        self.y_axis = Axis(data=np.linspace(1, self.map[ic4.PropId.HEIGHT].maximum, self.map.get_value_int(ic4.PropId.HEIGHT)))

        # Way to define a wait function with arguments
        #wait_func = lambda: self.wait_for_frame(since='lastread', timeout=20.0)
        #callback = ImagingSourceCallback(wait_func)

        #self.callback_thread = QtCore.QThread()  # creation of a Qt5 thread
        #callback.moveToThread(self.callback_thread)  # callback object will live within this thread
        #callback.data_sig.connect(self.emit_data)  # when the wait for acquisition returns (with data taken), emit_data will be fired
        #self.callback_signal.connect(callback.wait_for_acquisition)
        #self.callback_thread.callback = callback
        #self.callback_thread.start()

        info = "Imaging Source camera initialized"
        print("Imaging Source camera initialized successfully")
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        if self.controller.is_streaming:
            self.controller.stream_stop()
        self.controller.device_close()
        ic4.Library.exit()
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
    
        self.sink = ic4.SnapSink()
        self.sink.AllocationStrategy(10, 4, 6, 0)
        self.controller.stream_setup(self.sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

        #self.callback_signal.emit()  # will trigger the wait for acquisition
        self.emit_data()
        self.controller.stream_stop()

    def emit_data(self):
        """
            Fonction used to emit data obtained by callback.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        try:
            buffer = self.sink.snap_single(1000)
            image = buffer.numpy_copy()
            buffer.release()
            if image is not None:
                self.data_grabed_signal.emit([DataFromPlugins(name='DMK Camera',
                                                              data=[np.squeeze(image)],
                                                              dim=self.data_shape,
                                                              labels=[f'DMK_{self.data_shape}'])])


            # To make sure that timed events are executed in continuous grab mode
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def handle_device_lost():
        print("Device lost!")

    def device_lost(self):
        token = self.controller.event_add_device_lost(self.handle_device_lost(self.controller))
        self.controller.event_remove_device_lost(token)
        self.controller.stream_stop()
        self.controller.device_close()


    def callback(self):
        """optional asynchrone method called when the detector has finished its acquisition of data"""
        data_tot = self.controller.your_method_to_get_data_from_buffer()
        self.dte_signal.emit(DataToExport('myplugin',
                                          data=[DataFromPlugins(name='Mock1', data=data_tot,
                                                                dim='Data2D', labels=['label1'],
                                                                x_axis=self.x_axis,
                                                                y_axis=self.y_axis), ]))
    def stop(self):
        """Stop the current grab hardware wise if necessary"""

        return ''
    
    def wait_for_frame(self, since="lastread", timeout=20.0, period=1E-3):
        """
        Wait for a new camera frame.

        Parameters:
        - `since`: Defines the reference point for new frames. Can be:
            - `"lastread"`: Wait for a frame after the last read frame.
            - `"lastwait"`: Wait for a frame after the last `wait_for_frame` call.
            - `"now"`: Wait for a frame acquired after this function call.
        - `timeout`: Maximum waiting time (in seconds).
        - `period`: Polling interval (in seconds).

        Raises:
        - `TimeoutError`: If no frame is received within `timeout`.
        """
        start_time = time.time()

        if since not in {"lastread", "lastwait", "now"}:
            raise ValueError(f"Invalid 'since' value: {since}")

        if since == "lastwait":
            while time.time() - start_time < timeout:
                try:
                    image = self.sink.snap_single(1000)
                    if image is not None:
                        self.last_wait_frame = image
                        return image
                except ic4.IC4Exception:
                    pass
                time.sleep(period)
            raise TimeoutError("⏳ Frame acquisition timed out!")

        elif since == "lastread":
            while self.get_new_images_range() is None:
                self.wait_for_frame(since="lastwait", timeout=timeout)

        else:  # "now"
            last_img = self.get_new_images_range()
            while True:
                self.wait_for_frame(since="lastwait", timeout=timeout)
                new_img = self.get_new_images_range()
                if new_img and (last_img is None or new_img[1] > last_img):
                    return

    def get_new_images_range(self):
        """Returns the index of the latest frame (if any)."""
        try:
            image = self.sink.snap_single(1000)
            if image is not None:
                self.last_read_frame = image
                return (self.last_read_frame, self.last_read_frame)  # Simulating a range
        except ic4.IC4Exception:
            pass
        return None
    
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
    main(__file__)