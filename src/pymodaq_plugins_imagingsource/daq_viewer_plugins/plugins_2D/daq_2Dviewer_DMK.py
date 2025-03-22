import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from PyQt5.QtCore import pyqtSignal

from qtpy import QtWidgets, QtCore
from time import perf_counter

import imagingcontrol4 as ic4
import cv2

ic4.Library.init()

class ProcessAndDisplayListener(ic4.QueueSinkListener):
    # Listener to demonstrate processing and displaying received images

    def __init__(self, d: ic4.Display):
        self.display = d

    def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
        # Just accept whatever is passed
        return True

    def frames_queued(self, sink: ic4.QueueSink):
        # Get the new buffer from the sink
        buffer = sink.pop_output_buffer()

        # Create a numpy view onto the buffer
        # This view is only valid while the buffer itself exists,
        # which is guaranteed by them both not being passed out of this function
        buffer_wrap = buffer.numpy_wrap()

        # Blur the buffer in-place using a rather large kernel
        cv2.blur(buffer_wrap, (31, 31), buffer_wrap)

        # Write some text so that the user doesn't hopelessly try to focus the lens
        cv2.putText(
            buffer_wrap,
            "This image is blurred using OpenCV",
            (100, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
        )

        # Send the modified buffer to the display
        self.display.display_buffer(buffer)



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
        {'title': 'Camera model:', 'name': 'camera_name', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Image width', 'name': 'width', 'type': 'int', 'value': 1280, 'default': 1280, 'min': 640, 'max': 1280},
        {'title': 'Image height', 'name': 'height', 'type': 'int', 'value': 960, 'default': 960, 'min': 480, 'max': 960},
        {'title': 'Exposure time (ms)', 'name': 'exposure_time', 'type': 'float', 'value': 100.0, 'default': 100.0, 'min': 100.0, 'max': 30000000.0},
        {'title': 'Gain', 'name': 'gain', 'type': 'float', 'value': 34, 'default': 34, 'min': 34, 'max': 255}
        
    ]

    callback_signal = pyqtSignal()

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: ic4.Grabber = None
        self.model_name = ic4.DeviceEnum.devices()[0].model_name
        self.map = None
        self.sink = ic4.QueueSink([ic4.PixelFormat.BGR8], max_output_buffers=1)

        
        self.width = 1
        self.height = 1
        self.exposure_time = 1
        self.gain = 1


        self.x_axis = Axis(data=np.linspace(0, 1, self.width))
        self.y_axis = Axis(data=np.linspace(0, 1, self.height))

        self.data_shape = 'Data2D'
        self.callback_thread = None

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
        elif param.name() == "exposure_time":
            self.controller.device_property_map.set_value(ic4.PropId.EXPOSURE, param.value)
        elif param.name() == "gain":
            self.controller.device_property_map.set_value(ic4.PropId.GAIN, param.value)

        

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
        
        if not ic4.DeviceEnum.devices():
            raise Exception("No Imaging Source camera found")

        self.ini_detector_init(old_controller=controller,
                               new_controller=ic4.Grabber())
        
        self.controller.device_open(self.device_info)
        self.controller.stream_setup(self.sink, ic4.StreamSetupOption.DEFER_ACQUISITION_START)
        self.map = self.controller.device_property_map

        self.map.set_value(ic4.PropId.WIDTH, self.width)
        self.map.set_value(ic4.PropId.HEIGHT, self.height)
        self.map.set_value(ic4.PropId.EXPOSURE_TIME, self.exposure_time)
        self.map.set_value(ic4.PropId.GAIN, self.gain)

        self.x_axis = Axis(data=np.linspace(1, self.map[ic4.PropId.WIDTH].maximum, self.width))
        self.y_axis = Axis(data=np.linspace(1, self.map[ic4.PropId.HEIGHT].maximum, self.height))

        callback = ImagingSourceCallback(self.grabber, self.sink)

        self.callback_thread = QtCore.QThread()  # Create a separate Qt thread
        callback.moveToThread(self.callback_thread)  # Move callback object to thread

        callback.data_sig.connect(self.emit_data)  # When data is available, process it

        self.callback_signal.connect(callback.wait_for_acquisition)  # Connect signal to wait function
        self.callback_thread.callback = callback
        self.callback_thread.start()

        info = "Imaging Source camera initialized"
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        self.controller.device_close()
        self.controller = None  # Garbage collect the controller
        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""

    def _prepare_view(self):
        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs are changed"""

        mock_data = np.zeros((self.width, self.height))

        if self.width != 1 and self.height != 1:
            data_shape = 'Data2D'
        else:
            data_shape = 'Data1D'

        if data_shape != self.data_shape:
            self.data_shape = data_shape
            # init the viewers
            self.data_grabed_signal_temp.emit([DataFromPlugins(name='Thorlabs Camera',
                                                               data=[np.squeeze(mock_data)],
                                                               dim=self.data_shape,
                                                               labels=[f'ThorCam_{self.data_shape}'])])
            QtWidgets.QApplication.processEvents()    

    def grab_data(self, Naverage=1, **kwargs):
        """
        Grabs the data. Synchronous method (kinda).
        ----------
        Naverage: (int) Number of averaging
        kwargs: (dict) of others optionals arguments
        """
        try:
            # Warning, acquisition_in_progress returns 1,0 and not a real bool
            if not self.controller.is_acquisition_active:
                self.controller.acquisition_stop()
                self.controller.acquisition_start()

            # Then start the acquisition
            self.callback_signal.emit()  # will trigger the wait for acquisition

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

    def emit_data(self):
        """
            Fonction used to emit data obtained by callback.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        try:
            # Get  data from buffer
            frame = self.sink.pop_output_buffer()
            # Emit the frame.
            if frame is not None:  # happens for last frame when stopping camera
                self.data_grabed_signal.emit([DataFromPlugins(name='DMK Camera',
                                                              data=[np.squeeze(frame)],
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
        self.controller.acquisition_stop()  # when writing your own plugin replace this line
        self.emit_status(ThreadCommand('Update_Status', ['Some info you want to log']))
        return ''
    
class ImagingSourceCallback(QtCore.QObject):
    """Callback object for Imaging Source camera"""
    data_sig = QtCore.Signal()

    def __init__(self, grabber, sink):
        super().__init__()
        self.grabber = grabber  # Reference to the Grabber
        self.sink = sink  # Reference to the SnapSink

    def wait_for_acquisition(self):
        """Waits for the next frame and emits signal when ready"""
        try:
            image = self.sink.snap_single(1000)  # Wait up to 1000ms for a frame
            if image is not None:
                self.data_sig.emit()  # Notify that a new frame is available
        except ic4.IC4Exception as ex:
            print(f"Error in acquisition: {ex.message}")


if __name__ == '__main__':
    main(__file__)