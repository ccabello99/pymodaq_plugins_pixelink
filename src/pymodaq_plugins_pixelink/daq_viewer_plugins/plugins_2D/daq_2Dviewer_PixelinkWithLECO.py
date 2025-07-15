import numpy as np
from pixelinkWrapper import*
import os
import imageio as iio
from uuid6 import uuid7
import h5py
import json

import warnings
import numpy as np
# Suppress only NumPy RuntimeWarnings (bc of crosshair bug)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq_plugins_pixelink.hardware.pixelink import PixelinkCamera, get_info_for_all_cameras, TemperatureMonitor
from pymodaq_plugins_pixelink.resources.extended_publisher import ExtendedPublisher
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.data import Axis, DataFromPlugins, DataToExport
from pymodaq.control_modules.viewer_utility_classes import main, DAQ_Viewer_base, comon_parameters
from qtpy import QtWidgets, QtCore
from typing import Optional


class DAQ_2DViewer_PixelinkWithLECO(DAQ_Viewer_base):
    """ 
    
    * Tested with DMK 42BUC03/33GR0134 cameras.
    * Tested on PyMoDAQ version >= 5.0.2
    * Tested on Windows 11
    * Installation instructions: For this camera, you need to install the Imaging Source drivers, 
                                specifically "Device Driver for USB Cameras" and/or "Device Driver for GigE Cameras" in legacy software

    """

    live_mode_available = True

    devices = get_info_for_all_cameras()
    camera_list = [device["Name"] for device in devices]

    
    # Default place to store qsettings for this module
    settings_pixelink = QtCore.QSettings("PyMoDAQ", "Pixelink")
    

    params = comon_parameters + [
        {'title': 'Camera List:', 'name': 'camera_list', 'type': 'list', 'value': '', 'limits': camera_list},
        {"title": "Device Info", "name": "device_info", "type": "group","children": [
            {"title": "Device Name", "name": "Name", "type": "str", "value": ""},
            {"title": "Model Name", "name": "Model_Name", "type": "str", "value": "", "readonly": True},
            {"title": "Serial Number", "name": "Serial_Number", "type": "str", "value": "", "readonly": True},
            {"title": "Firmware Version", "name": "Firmware_Version", "type": "str", "value": "", "readonly": True}
        ]},        
        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
            {'title': 'Image Width', 'name': 'width', 'type': 'int', 'value': 1024, 'readonly': True},
            {'title': 'Image Height', 'name': 'height', 'type': 'int', 'value': 768, 'readonly': True},
        ]},
        {'title': 'LECO Logging', 'name': 'leco_log', 'type': 'group', 'children': [
            {'title': 'Send Frame Data ?', 'name': 'leco_send', 'type': 'led_push', 'value': False, 'default': False,
                'tip': 'This leads to huge performance drop as of now. Only use for single grabs, not continuous'},
            {'title': 'Publisher Name', 'name': 'publisher_name', 'type': 'str', 'value': ''},
            {'title': 'Proxy Server Address', 'name': 'proxy_address', 'type': 'str', 'value': 'localhost', 'default': 'localhost',
                'tip': 'Either IP or hostname of LECO proxy server'},
            {'title': 'Proxy Server Port', 'name': 'proxy_port', 'type': 'int', 'value': 11100, 'default': 11100},
            {'title': 'Metadata', 'name': 'leco_metadata', 'type': 'str', 'value': '', 'readonly': True},
            {'title': 'Saving Base Path:', 'name': 'leco_basepath', 'type': 'browsepath', 'value': '', 'filetype': False,
                'tip': 'This is the base directory for a file path sent from a remote director in the metadata'},
        ]}
    ]

    def ini_attributes(self):
        """Initialize attributes"""

        self.controller: None
        self.user_id = None

        self.data_shape = None
        self.save_frame = False

        # For LECO operation
        self.metadata = None
        self.data_publisher = None
        self.send_frame_leco = False

    def init_controller(self) -> PixelinkCamera:

        # Init camera
        self.user_id = self.settings.param('camera_list').value()
        self.emit_status(ThreadCommand('Update_Status', [f"Trying to connect to {self.user_id}", 'log']))
        devices = get_info_for_all_cameras()
        camera_list = [device["Name"] for device in devices]
        for cam in camera_list:
            if cam == self.user_id:
                device_idx = camera_list.index(self.user_id)
                device_info = devices[device_idx]
                return PixelinkCamera(info=device_info, callback=self.emit_data_callback)
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

        self.ini_detector_init(old_controller=controller,
                            new_controller=self.init_controller())

        # Initialize pixel format before starting stream
        ret = PxLApi.setFeature(self.controller.camera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, [0])
        if not PxLApi.apiSuccess(ret[0]):
            print("ERROR setting pixel format: {0}".format(ret[0]))
            print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReturnCode)

        # Update the UI with available and current camera parameters
        self.add_attributes_to_settings()
        self.update_params_ui()
        for param in self.settings.children():
            if param.name() == 'device_info':
                continue
            param.sigValueChanged.emit(param, param.value())
            if param.hasChildren():
                for child in param.children():
                    child.sigValueChanged.emit(child, child.value())

        # Setup data publisher for LECO if data publisher name is set (ideally it should match the LECO actor name)
        publisher_name = self.settings.child('leco_log', 'publisher_name').value()
        proxy_address = self.settings.child('leco_log', 'proxy_address').value()
        proxy_port = self.settings.child('leco_log', 'proxy_port').value()
        if publisher_name == '':
            print("Publisher name is not set ! Set this first and then reinitialize for LECO logging.")
            self.emit_status(ThreadCommand('Update_Status', ["Publisher name is not set ! Set this first and then reinitialize for LECO logging."]))
        else:
            self.data_publisher = ExtendedPublisher(full_name=publisher_name, host=proxy_address, port=proxy_port)
            print(f"Data publisher {publisher_name} initialized for LECO logging")
            self.emit_status(ThreadCommand('Update_Status', [f"Data publisher {publisher_name} initialized for LECO logging"]))

        
        try:
            base_path = self.settings_pixelink.value('leco_log/basepath', os.path.join(os.path.expanduser('~'), 'Downloads'))
        except Exception as e:
            print(f"Error finding LECO base path: {e}")
            base_path = ''
        self.settings.child('leco_log', 'leco_basepath').setValue(base_path)

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
        name = param.name()
        value = param.value()

        if name == "camera_list":
            if self.controller != None:
                self.close()
            self.ini_detector()

        if name == "device_state_save":
            if value:
                self.controller.save_device_state()
                self.settings.param('device_state_save').setValue(False)
                param.sigValueChanged.emit(param, False)
                self.emit_status(ThreadCommand('Update_Status', ["Camera state saved successfully."]))
            else:
                pass
            return
        if name == "Name":
            self.user_id = value
            PxLApi.setCameraName(self.controller.camera, value)
            self.close()
            # Update camera list to account for new device name & re-init
            devices = get_info_for_all_cameras()
            camera_list = [device["Name"] for device in devices]
            param = self.settings.param('camera_list')
            param.setValue(self.user_id)
            param.setLimits(camera_list)
            param.sigValueChanged.emit(param, value)
            param.sigLimitsChanged.emit(param, camera_list)
            self.ini_detector()
            print("Camera name updated successfully. Restart live grab !")
            self.emit_status(ThreadCommand('Update_Status', ["Camera name updated successfully. Restart live grab !"]))
            return
        
        # Special cases
        if name == 'SHUTTER':
            feature_id = self.controller.feature_map[name]["id"]
            value *= 1e-3
            ret = PxLApi.setFeature(self.controller.camera, feature_id, PxLApi.FeatureFlags.MANUAL, [value])
            feature_id = self.controller.feature_map["FRAME_RATE"]["id"]
            param_index = self.controller.feature_map["FRAME_RATE"]["params"]["VALUE"]
            new_frame_rate = PxLApi.getFeature(self.controller.camera, feature_id)[2][0]
            ret = PxLApi.getCameraFeatures(self.controller.camera, feature_id)
            feature = ret[1].Features[0]
            min_limit = feature.Params[param_index].fMinValue
            max_limit = feature.Params[param_index].fMaxValue
            new_limits = [min_limit, max_limit]
            # Update frame rate to be compatible w/ new exposure time
            param = self.settings.param('FRAME_RATE')
            param.setValue(new_frame_rate)
            param.setLimits(new_limits)
            param.sigValueChanged.emit(param, new_frame_rate)
            param.sigLimitsChanged.emit(param, new_limits)
            return
        if name == 'PIXEL_FORMAT':
            self.controller.stop_acquisition()
            if value == 'Mono8':
                ret = PxLApi.setFeature(self.controller.camera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, [PxLApi.PixelFormat.MONO8])
            elif value == 'Mono12Packed':
                ret = PxLApi.setFeature(self.controller.camera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, [PxLApi.PixelFormat.MONO12_PACKED])
            elif value == 'Mono16':
                ret = PxLApi.setFeature(self.controller.camera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, [PxLApi.PixelFormat.MONO16])
            if not PxLApi.apiSuccess(ret[0]):
                print("ERROR setting pixel format: {0}".format(ret[0]))
                print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReturnCode)
            self.controller.start_acquisition()
            self._prepare_view()
            return
        if name == 'MODE':
            if value:
                if self.settings.child('trigger','POLARITY').value() == 'Rising Edge':
                    polarity = PxLApi.Polarity.ACTIVE_HIGH
                elif self.settings.child('trigger', 'POLARITY').value() == 'Falling Edge':
                    polarity = PxLApi.Polarity.ACTIVE_LOW
                self.controller.stop_acquisition() # Make sure stream is stopped before set trigger
                self.controller.set_triggering(
                    PxLApi.TriggerModes.MODE_0,
                    PxLApi.TriggerTypes.HARDWARE,
                    polarity,
                    self.settings.child('trigger', 'DELAY').value()*1e-6,
                    0)
                self.controller.start_acquisition() # Turn stream back on
            else:
                self.save_frame = False
                param = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
                param.setValue(False) # Turn off save on trigger if we turn off triggering
                param.sigValueChanged.emit(param, False)
                self.controller.disable_triggering()
        if name == 'TriggerSave':
            if not self.settings.child('trigger', 'MODE').value():
                print("Trigger mode is not active ! Start triggering first !")
                self.emit_status(ThreadCommand('Update_Status', ["Trigger mode is not active ! Start triggering first !"]))
                param = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
                param.setValue(False) # Turn off save on trigger if triggering is off
                param.sigValueChanged.emit(param, False) 
                return
            if value:
                self.save_frame = True
                return
            else:
                self.save_frame = False
                return
        # we only need to reference these, nothing to do with the cam
        if name == 'TriggerSaveLocation':
            return
        if name == 'TriggerSaveIndex':
            return
        if name == 'Filetype':
            return
        if name == 'Prefix':
            return
        if name == 'TEMPERATURE_MONITOR':
            if value:
                # Start thread for camera temp. monitoring
                self.start_temperature_monitoring()
            else:
                # Stop background threads
                self.stop_temp_monitoring()
            return
        if name == 'leco_send':
            if value:
                self.send_frame_leco = True
            else:
                self.send_frame_leco = False
            return
        if name == 'leco_basepath':
            base_path = value
            if not os.path.exists(base_path):
                print(f"LECO saving base path {base_path} does not exist !")
                self.emit_status(ThreadCommand('Update_Status', [f"LECO saving base path {base_path} does not exist !"]))
            else:
                try:
                    self.settings_basler.setValue('leco_log/basepath', base_path)
                    print(f"LECO saving base path set to {base_path}")
                    self.emit_status(ThreadCommand('Update_Status', [f"LECO saving base path set to {base_path}"]))
                except Exception as e:
                    print(f"Error setting LECO saving base path: {e}")
                    self.emit_status(ThreadCommand('Update_Status', [f"Error setting LECO saving base path: {e}"]))
        if name == 'leco_metadata':
            self.metadata = json.loads(value)       
    
        # Update other features
        if name in self.controller.attribute_names:
            feature_id = self.controller.feature_map[name]["id"]
            param_index = self.controller.feature_map[name]["params"]["VALUE"]
            try:
                ret = PxLApi.setFeature(self.controller.camera, feature_id, PxLApi.FeatureFlags.MANUAL, [value])
                if not PxLApi.apiSuccess(ret[0]):
                    print("ERROR setting feature {name}: {0}".format(ret[0]))
                    print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReturnCode)
                    self.emit_status(ThreadCommand('Update_Status', ["Error message:", PxLApi.getErrorReport(ret[0])[1].strReturnCode]))
            except Exception:
                pass
    
    def _prepare_view(self):
        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs are changed"""
        feature_id = self.controller.feature_map['ROI']['id']

        width = int(PxLApi.getFeature(self.controller.camera, feature_id)[2][2])
        height = int(PxLApi.getFeature(self.controller.camera, feature_id)[2][3])

        self.settings.child('roi', 'width').setValue(width)
        self.settings.child('roi', 'height').setValue(height)

        mock_data = np.zeros((height, width))

        self.x_axis = Axis(label='Pixels', data=np.linspace(1, width, width), index=1)

        if width != 1 and height != 1:
            data_shape = 'Data2D'
            self.y_axis = Axis(label='Pixels', data=np.linspace(1, height, height), index=0)
            self.axes = [self.y_axis, self.x_axis]
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

    def grab_data(self, Naverage: int = 1, live: bool = False, **kwargs) -> None:
        try:
            self._prepare_view()
            if live:
                self.controller.start_acquisition()
            else:
                self.controller.start_acquisition()
                while not self.controller.listener.frame_ready:
                    pass # do nothing until a frame is ready
                self.controller.stop_acquisition()
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))


    def emit_data_callback(self, frame_data: dict) -> None:
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(-1)
        shape = frame.shape

        # First emit data to the GUI
        dte = DataToExport(f'{self.user_id}', data=[DataFromPlugins(
            name=f'{self.user_id}',
            data=[np.squeeze(frame)],
            dim=self.data_shape,
            labels=[f'{self.user_id}_{self.data_shape}'],
            axes=self.axes)])
        self.dte_signal.emit(dte)

        # Now, handle data saving with filepath given by user in trigger save settings or from metadata set remotely with LECO
        self.handle_metadata_and_saving(frame, timestamp, shape)

        # Prepare for next frame
        self.metadata = None
        self.controller.listener.frame_ready = False

    def stop(self):
        self.controller.stop_acquisition()
        return ''
    
    def handle_metadata_and_saving(self, frame, timestamp, shape):
        metadata = self.get_metadata_and_save(frame, timestamp, shape)
        if self.send_frame_leco:
            self.publish_metadata(metadata, frame)
        else:
            self.publish_metadata(metadata)   
    
    def get_metadata_and_save(self, frame, timestamp, shape):
        if self.save_frame:
            if self.metadata is not None:
                metadata = self.metadata
                metadata['burst_metadata']['user_id'] = self.user_id
            else:
                metadata = {'burst_metadata':{}, 'file_metadata': {}, 'detector_metadata': {}}
                metadata['burst_metadata']['uuid'] = str(uuid7())
                metadata['burst_metadata']['user_id'] = self.user_id
                metadata['burst_metadata']['timestamp'] = timestamp

            # Account for some uncertainty in timestamp of frame, assume 1 ms for now
            metadata['detector_metadata']['fuzziness'] = 1
            count = 0
            for name in self.controller.attribute_names:
                if name == 'GAIN':
                    metadata['detector_metadata']['gain'] = self.settings.child('gain', name).value()
                    count += 1
                if name == 'SHUTTER':
                    metadata['detector_metadata']['exposure_time'] = self.settings.child('exposure', name).value()
                    count += 1
                if count == 2:
                    break
            metadata['detector_metadata']['shape'] = shape
            index = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSaveIndex')
            filetype = self.settings.child('trigger', 'TriggerSaveOptions', 'Filetype').value()
            if self.metadata is not None:
                metadata = self.metadata
                filepath = self.metadata['file_metadata']['filepath']
                filename = self.metadata['file_metadata']['filename']
                self.metadata['burst_metadata']['user_id'] = self.user_id
                basepath = self.settings.child('leco_log', 'leco_basepath').value()
                filepath = os.path.normpath(os.path.join(basepath, filepath.lstrip(os.path.sep)))
            else:
                filepath = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSaveLocation').value()
                prefix = self.settings.child('trigger', 'TriggerSaveOptions', 'Prefix').value()
                if not filepath:
                    filepath = os.path.join(os.path.expanduser('~'), 'Downloads')
                filename = f"{prefix}{index.value()}.{filetype}"
                metadata = {'burst_metadata':{}, 'file_metadata': {}, 'detector_metadata': {}}
                metadata['burst_metadata']['uuid'] = str(uuid7())
                metadata['burst_metadata']['user_id'] = self.user_id
                metadata['burst_metadata']['timestamp'] = timestamp
                metadata['file_metadata']['filepath'] = filepath
                metadata['file_metadata']['filename'] = filename
                index.setValue(index.value()+1)
                index.sigValueChanged.emit(index, index.value())
            # Include device metadata to send back
            # Account for some uncertainty in timestamp of frame, assume 1 ms for now
            metadata['detector_metadata']['fuzziness'] = 1
            count = 0
            for name in self.controller.attribute_names:
                if name == 'GAIN':
                    metadata['detector_metadata']['gain'] = self.settings.child('gain', name).value()
                    count += 1
                if name == 'SHUTTER':
                    metadata['detector_metadata']['exposure_time'] = self.settings.child('exposure', name).value()
                    count += 1
                if count == 2:
                    break
            metadata['detector_metadata']['shape'] = shape
            if filetype == 'h5':
                if not filename.endswith('.h5'):
                    filename += '.h5'
                full_path = os.path.join(filepath, filename)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with h5py.File(full_path, 'w') as f:
                    dataset_name = f"frame_{timestamp}"
                    f.create_dataset(dataset_name, data=frame)
                    f.attrs['uuid'] = metadata['burst_metadata']['uuid']
                    f.attrs['user_id'] = metadata['burst_metadata']['user_id']
                    f.attrs['timestamp'] = timestamp
                    f.attrs['exposure_time'] = metadata['detector_metadata']['exposure_time']
                    f.attrs['gain'] = metadata['detector_metadata']['gain']
                    f.attrs['shape'] = metadata['detector_metadata']['shape']
                    f.attrs['fuzziness'] = metadata['detector_metadata']['fuzziness']
                    f.attrs['format_version'] = 'hdf5-v0.1'
            else:
                if not filename.endswith(f".{filetype}"):
                    filename += f".{filetype}"
                if filetype not in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
                    print(f"Unsupported file type {filetype} for saving frame. Supported types are: png, jpg, jpeg, tiff, tif, h5")
                    self.emit_status(ThreadCommand('Update_Status', [f"Unsupported file type {filetype} for saving frame. Supported types are: png, jpg, jpeg, tiff, tif, h5"]))
                    return
                full_path = os.path.join(filepath, f"{filename}.{filetype}")
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                iio.imwrite(full_path, frame)
        return metadata
        
    def publish_metadata(self, metadata, frame: Optional[np.ndarray] = None):
        if self.data_publisher is not None and self.save_frame:
            if self.send_frame_leco:                        
                self.data_publisher.send_data2({self.settings.child('leco_log', 'publisher_name').value(): 
                                                {'frame': frame, 'metadata': metadata, 
                                                 'message_type': 'detector', 
                                                 'serial_number': self.controller.device_info.GetSerialNumber(),
                                                 'format_version': 'hdf5-v0.1'}})
            else:
                self.data_publisher.send_data2({self.settings.child('leco_log', 'publisher_name').value(): 
                                                {'metadata': metadata, 
                                                 'message_type': 'detector',
                                                 'serial_number': self.controller.device_info.GetSerialNumber(),
                                                 'format_version': 'hdf5-v0.1'}})
    
    def close(self):
        """Terminate the communication protocol"""
        self.controller.attributes = None
        self.controller.close()
            
        # Stop any background threads
        if hasattr(self, 'listener'):
            self.listener.stop_listener()
        self.stop_temp_monitoring()

        # Make sure we set these to false if camera disconnected
        param = self.settings.child('trigger', 'MODE')
        param.setValue(False) # Turn off save on trigger if triggering is off
        param.sigValueChanged.emit(param, False)
        param = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
        param.setValue(False) # Turn off save on trigger if triggering is off
        param.sigValueChanged.emit(param, False)          

        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""
        print(f"{self.user_id} communication terminated successfully")
        self.emit_status(ThreadCommand('Update_Status', [f"{self.user_id} communication terminated successfully"]))
    
    def roi_select(self, roi_info, ind_viewer):
        self.roi_info = roi_info
    
    def crosshair(self, crosshair_info, ind_viewer=0):
        sleep_ms = 150
        self.crosshair_info = crosshair_info
        QtCore.QTimer.singleShot(sleep_ms, QtWidgets.QApplication.processEvents)

    def start_temperature_monitoring(self):
        self.temp_thread = QtCore.QThread()
        self.temp_worker = TemperatureMonitor(self.controller.camera)

        self.temp_worker.moveToThread(self.temp_thread)

        self.temp_thread.started.connect(self.temp_worker.run)
        self.temp_worker.temperature_updated.connect(self.on_temperature_update)
        self.temp_worker.finished.connect(self.temp_thread.quit)
        self.temp_worker.finished.connect(self.temp_worker.deleteLater)
        self.temp_thread.finished.connect(self.temp_thread.deleteLater)

        self.temp_thread.start()

    def stop_temp_monitoring(self):
        if hasattr(self, 'temp_worker') and self.temp_worker is not None:
            self.temp_worker.stop()
            self.temp_worker = None
        if hasattr(self, 'temp_thread') and self.temp_thread is not None:
            try:
                self.temp_thread.quit()
                self.temp_thread.wait()
            except RuntimeError:
                pass  # Already deleted
            self.temp_thread = None
        # Make sure temp. monitoring param is false in GUI
        param = self.settings.child('temperature', 'TEMPERATURE_MONITOR')
        param.setValue(False)
        param.sigValueChanged.emit(param, param.value())


    def on_temperature_update(self, temp: float):
        param = self.settings.child('temperature', 'SENSOR_TEMPERATURE')
        param.setValue(temp)
        param.sigValueChanged.emit(param, temp)
        if temp > 60:
            self.emit_status(ThreadCommand('Update_Status', [f"WARNING: {self.user_id} camera is hot !!"]))


    def add_attributes_to_settings(self):
        existing_group_names = {child.name() for child in self.settings.children()}

        for attr in self.controller.attributes:
            attr_name = attr['name']
            if attr.get('type') == 'group':
                if attr_name not in existing_group_names:
                    self.settings.addChild(attr)
                else:
                    group_param = self.settings.child(attr_name)

                    existing_children = {child.name(): child for child in group_param.children()}

                    expected_children = attr.get('children', [])
                    for expected in expected_children:
                        expected_name = expected['name']
                        if expected_name not in existing_children:
                            for old_name, old_child in existing_children.items():
                                if old_child.opts.get('title') == expected.get('title') and old_name != expected_name:
                                    self.settings.child(attr_name, old_name).show(False)
                                    break

                            group_param.addChild(expected)
            else:
                if attr_name not in existing_group_names:
                    self.settings.addChild(attr)
        
    
    def update_params_ui(self):

        # Common syntax for any camera model
        self.settings.child('device_info','Name').setValue(self.controller.device_info["Name"])
        self.settings.child('device_info','Model_Name').setValue(self.controller.device_info["Model Name"])
        self.settings.child('device_info','Serial_Number').setValue(self.controller.device_info["Serial Number"])
        self.settings.child('device_info','Firmware_Version').setValue(self.controller.device_info["Firmware Version"])

        for param in self.controller.attributes:
            param_type = param['type']
            param_name = param['name']
            
            # Skip these
            if param_name == "device_info":
                continue
            if param_name == "device_state_save":
                continue
            if param_name == "trigger":
                continue
            if param_name == "temperature":
                continue

            if param_type == 'group':
                # Recurse over children in groups
                for child in param['children']:
                    child_name = child['name']
                    child_type = child['type']
                    
                    # Get feature ID and parameter index
                    feature_id = self.controller.feature_map[child_name]["id"]
                    param_index = self.controller.feature_map[child_name]["params"]["VALUE"]

                    # Now use those for queries
                    ret = PxLApi.getFeature(self.controller.camera, feature_id)
                    if not PxLApi.apiSuccess(ret[0]):
                        print("ERROR getting feature: {0}".format(ret[0]))
                        print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReturnCode)
                    value = ret[2][0]

                    # Special case: if parameter is SHUTTER (exposure time), convert to ms from s
                    if child_name == 'SHUTTER':
                        value *= 1e3

                    # Special case: if parameter is Pixel Format, initialize with Mono12Packed:
                    if child_name == 'PIXEL_FORMAT':
                        ret = PxLApi.setFeature(self.controller.camera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, [PxLApi.PixelFormat.MONO12_PACKED])
                        continue


                    # Set the value
                    self.settings.child(param_name, child_name).setValue(value)

                    # Set limits if defined
                    if 'limits' in child and child_type in ['float', 'slide', 'int'] and not child.get('readonly', False):
                        try:
                            ret = PxLApi.getCameraFeatures(self.controller.camera, feature_id)
                            feature = ret[1].Features[0]
                            min_limit = feature.Params[param_index].fMinValue
                            max_limit = feature.Params[param_index].fMaxValue

                            if child_name == 'SHUTTER':
                                min_limit *= 1e3
                                max_limit *= 1e3

                            self.settings.child(param_name, child_name).setLimits([min_limit, max_limit])
                        except Exception:
                            pass
            else:

                # Get feature ID and parameter index
                feature_id = self.controller.feature_map[param_name]["id"]
                param_index = self.controller.feature_map[param_name]["params"]["VALUE"]

                # Now use those for queries
                ret = PxLApi.getFeature(self.controller.camera, feature_id)
                if not PxLApi.apiSuccess(ret[0]):
                    print("ERROR getting feature: {0}".format(ret[0]))
                    print("Error message:", PxLApi.getErrorReport(ret[0])[1].strReturnCode)
                value = ret[2][0]

                # Set the value
                self.settings.param(param_name).setValue(value)

                if 'limits' in param and param_type in ['float', 'slide', 'int'] and not param.get('readonly', False):
                    try:
                        ret = PxLApi.getCameraFeatures(self.controller.camera, feature_id)
                        feature = ret[1].Features[0]
                        min_limit = feature.Params[param_index].fMinValue
                        max_limit = feature.Params[param_index].fMaxValue

                        self.settings.param(param_name).setLimits([min_limit, max_limit])

                    except Exception:
                        pass

if __name__ == '__main__':
    main(__file__, init=False)