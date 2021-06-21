# TFG_UB_Fisica_SergiSelles_ENG-ELECTRONICA-TELECOMUNICACIO
Code corresponding to TFG: System of Computer Vision and Deep Learning for detection and tracking of a person for Activity Monitoring.

CoralAI_detector: 
  - Code to run on hostcomputer with Windows 10, Coral AI USB and Intel RealSense D435i camera connected.
  - Install all dependencies corresponding to the libraries in the code.
  - Call the script using example indicated in CoralAI_detector.py file.

Nano_detector: 
  - Code to run on NVIDIA Jetson Nano computer with Ubuntu 18.04.
  - Install microSD card image with Python and Deep Learning libraries like JetPack 4.5.1 (https://developer.nvidia.com/embedded/jetpack). 
  - Complete installation of all remain libraries indicated in the code.
  - Download to ./model folder the model ssd_mobilenet_v1_coco_trt.pb from gdrive link https://drive.google.com/file/d/1tz8qohnGKe1kcufhIt6EX6_dc18cm_he/view?usp=sharing. 
  - Call the script using example indicated in Nano_detector.py file.
