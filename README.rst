pymodaq_plugins_pixelink
###########################################

.. the following must be adapted to your developed package, links to pypi, github  description...

.. image:: https://img.shields.io/pypi/v/pymodaq_plugins_pixelink.svg
   :target: https://pypi.org/project/pymodaq_plugins_pixelink/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/ccabello99/pymodaq_plugins_pixelink/workflows/Upload%20Python%20Package/badge.svg
   :target: https://github.com/ccabello99/pymodaq_plugins_pixelink
   :alt: Publication Status

.. image:: https://github.com/ccabello99/pymodaq_plugins_pixelink/actions/workflows/Test.yml/badge.svg
    :target: https://github.com/ccabello99/pymodaq_plugins_pixelink/actions/workflows/Test.yml


PyMoDAQ plugin for interfacing with Pixelink Cameras


Authors
=======

* First Author  (christian.cabello@ip-paris.fr)


Instruments
===========

Below is the list of instruments included in this plugin

Actuators
+++++++++

Viewer0D
++++++++

Viewer1D
++++++++


Viewer2D
++++++++

* **Pixelink**: control of Pixelink cameras


PID Models
==========


Extensions
==========


Installation instructions
=========================
* Tested on PyMoDAQ version 5.0.5
* Tested on Windows 11
* Must install Pixelink Capture or Pixelink SDK before use. If you already have a 32 bit install on your PC, uninstall this and reinstall the 64 bit version.
* Config files are needed for different camera models. Example for the PL-B953U camera is given in the resources directory. The name of the config file should be config_<model_name> where model_name is the output of getCameraInfo(cam)[1].ModelName. The module will look for this file in the ProgramData/.pymodaq folder in Windows and if not found, a default config file can be created upon camera initialization.
