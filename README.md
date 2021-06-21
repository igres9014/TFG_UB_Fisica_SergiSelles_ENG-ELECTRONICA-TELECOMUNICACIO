# TFG_UB_Fisica_SergiSelles_ENG-ELECTRONICA-TELECOMUNICACIO
Code corresponding to TFG: System of Computer Vision and Deep Learning for detection and tracking of a person for Activity Monitoring.

- REAL-TIME 3D detection and tracking of a person.

- Code for two different implementations: 
    - (1) Laptop Windows 10 + Coral AI USB Accelerator + Intel RealSense D435i camera
    - (2) NVIDIA Jetson NANO + Intel RealSense D435i camera


CoralAI_detector: 
  - Code to run on laptop with Windows 10, Coral AI USB and Intel RealSense D435i camera connected.
  - Install all dependencies corresponding to the libraries indicated in CoralAI_detector.py
  - Call the script as the example provided in the code.

Nano_detector: 
  - Code to run on NVIDIA Jetson Nano computer with Ubuntu 18.04 and Intel RealSense D435i camera connected.
  - Install microSD card image with Python and Deep Learning libraries like JetPack 4.5.1 https://developer.nvidia.com/embedded/jetpack
  - Complete installation of all remain libraries in Nano_detector.py
  - Download to ./model folder the model ssd_mobilenet_v1_coco_trt.pb from gdrive link: https://drive.google.com/file/d/1tz8qohnGKe1kcufhIt6EX6_dc18cm_he/view?usp=sharing
  - Call the script as the example provided in the code.

Example of real-time streaming output:

![image](https://user-images.githubusercontent.com/72890882/122738798-9a39e900-d282-11eb-8734-b4c0e56a8017.png)


![image](https://user-images.githubusercontent.com/72890882/122737767-aec9b180-d281-11eb-8dfc-6b0c775d4ce8.png)
