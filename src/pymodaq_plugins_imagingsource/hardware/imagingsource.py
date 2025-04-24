import logging
from typing import Any, Callable, List, Optional, Tuple, Union

from numpy.typing import NDArray
import imagingcontrol4 as ic4
import numpy as np
from qtpy import QtCore
import json
import os
from PyQt6.QtCore import pyqtSignal

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


pixel_lengths: dict[str, float] = {
    # camera model name: pixel length in µm
    "daA1280-54um": 3.75,
    "daA2500-14um": 2.2,
    "daA3840-45um": 2,
    "acA640-120gm": 5.6,
    "acA645-100gm": 5.6,
    "acA1920-40gm": 5.86,
}


class ImagingSourceCamera:
    """Control a Imaging Source camera in the style of pylablib.

    It wraps an :class:`pylon.InstantCamera` instance.

    :param name: Full name of the device.
    :param callback: Callback method for each grabbed image
    """

    tlFactory: ic4.Library
    camera: ic4.Grabber
    sink: ic4.QueueSink

    def __init__(self, info: str, callback: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        # create camera object

        self.camera = ic4.Grabber()
        self.model_name = info.model_name
        self.device_info = info

        self.gain_value = None
        self.gain_auto = None
        self.exposure_time = None
        self.exposure_auto = None
        self.gamma = None
        self.frame_rate = None
        self.gevscpd = None

        # register configuration event handler
        self.configurationEventHandler = ConfigurationHandler()

        # Callback setup for image grabbing
        self.listener = Listener()
        self.sink = ic4.QueueSink(self.listener, max_output_buffers=1)

        self._pixel_length: Optional[float] = None
        self.attributes = {}
        self.open()
        if callback is not None:
            self.set_callback(callback=callback)

    def open(self) -> None:
        self.camera.device_open(self.device_info)
        self.get_attributes()

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
        model_name = self.model_name.replace(" ", "-")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f'config_{model_name}.json')
        with open(file_path, 'r') as file:
            self.attributes = json.load(file)
            self.gain_value = self.attributes["Gain"]["name"]
            self.gain_auto = self.attributes["Gain Auto"]["name"]
            self.exposure_time = self.attributes["Exposure Time"]["name"]
            self.exposure_auto = self.attributes["Exposure Auto"]["name"]
            self.gamma = self.attributes["Gamma"]["name"]
            self.frame_rate = self.attributes["Acquisition Frame Rate"]["name"]
            self.brightness = self.attributes["Brightness"]["name"]
            self.contrast = self.attributes["Contrast"]["name"]


    def get_roi(self) -> Tuple[float, float, float, float, int, int]:
        """Return x0, width, y0, height, xbin, ybin."""
        x0 = self.camera.device_property_map.get_value_int('OffsetX')
        width = self.camera.device_property_map.get_value_int('Width')
        y0 = self.camera.device_property_map.get_value_int('OffsetY')
        height = self.camera.device_property_map.get_value_int('Height')
        xbin = self.camera.device_property_map.get_value_int('BinningHorizontal')
        ybin = self.camera.device_property_map.get_value_int('BinningVertical')
        return x0, x0 + width, y0, y0 + height, xbin, ybin

    def set_roi(
        self, hstart: int, hend: int, vstart: int, vend: int, hbin: int, vbin: int
    ) -> None:
        m_width, m_height = self.get_detector_size()
        inc = self.camera.device_property_map['Width'].increment  # minimum step size
        hstart = detector_clamp(hstart, m_width) // inc * inc
        vstart = detector_clamp(vstart, m_height) // inc * inc
        self.camera.device_property_map.try_set_value('Width', int((detector_clamp(hend, m_width) - hstart) // inc * inc))
        self.camera.device_property_map.try_set_value('Height', int((detector_clamp(vend, m_height) - vstart) // inc * inc))
        self.camera.device_property_map.try_set_value('BinningHorizontal', int(hbin))
        self.camera.device_property_map.try_set_value('BinningVertical', int(vbin))

    def get_detector_size(self) -> Tuple[int, int]:
        """Return width and height of detector in pixels."""
        return self.camera.device_property_map['Width'].maximum, self.camera.device_property_map['Height'].maximum
    
    def clear_acquisition(self):
        """Stop acquisition"""
        pass

    def setup_acquisition(self) -> None:
        self.camera.stream_setup(self.sink, setup_option=ic4.StreamSetupOption.DEFER_ACQUISITION_START)

    def close(self) -> None:
        if self.camera.is_acquisition_active:
            self.camera.acquisition_stop()
        if self.camera.is_streaming:
            self.camera.stream_stop()
        self.camera.device_close()
        self._pixel_length = None

    def start_grabbing(self, frame_rate: int) -> None:
        """Start continuously to grab data.

        Whenever a grab succeeded, the callback defined in :meth:`set_callback` is called.
        """
        try:
            self.camera.device_property_map.set_value(ic4.PropId.ACQUISITION_FRAME_RATE, frame_rate)
        except ic4.IC4Exception:
            pass
        self.camera.acquisition_start()

    @property
    def pixel_length(self) -> float:
        """Get the pixel length of the camera in µm.

        Returns None if the pixel length of the specific model is not known
        """
        if self._pixel_length is None:
            try:
                self._pixel_length = pixel_lengths[self.model_name]
            except KeyError:
                self._pixel_length = None
        return self._pixel_length

    @pixel_length.setter
    def pixel_length(self, value):
        self._pixel_length = value

class ConfigurationHandler:
    """Handle the configuration events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals = self.ConfigurationHandlerSignals()

    class ConfigurationHandlerSignals(QtCore.QObject):
        """Signals for the CameraEventHandler."""

        cameraRemoved = pyqtSignal(object)

    def OnOpened(self, camera: ic4.Grabber) -> None:
        """Standard configuration after being opened."""
        #camera.PixelFormat.SetValue("Mono12")
        #camera.GainAuto.SetValue("Off")
        #camera.ExposureAuto.SetValue("Off")
        pass

    def event_add_device_lost(handler: Callable[[ic4.Grabber], None]) -> ic4.Grabber.DeviceLostNotificationToken:
        """Emit a signal that the camera is removed."""
        #self.signals.cameraRemoved.emit(camera)
        pass


class Listener(ic4.QueueSinkListener):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals = self.ListenerSignal()
        self.frame_ready = False

    def frames_queued(self, sink: ic4.QueueSink):
        buffer = sink.try_pop_output_buffer()
        if buffer is not None:
            self.frame_ready = True
            frame = buffer.numpy_copy()
            buffer.release()
            self.signals.data_ready.emit(frame)
            

    def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
        return True

    def sink_disconnected(self, sink: ic4.QueueSink):
        pass

    class ListenerSignal(QtCore.QObject):
        """Signals for the ImageEventHandler."""

        data_ready = pyqtSignal(object)


def detector_clamp(value: Union[float, int], max_value: int) -> int:
    """Clamp a value to possible detector position."""
    return max(0, min(int(value), max_value))