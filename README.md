# MIA_Hand_XAI_Edge_AI_Simulation

# Semi-Autonomous Prosthetic Vision System

This project investigates a vision-based perception system for semi-autonomous upper-limb prostheses. The main goal is to reduce the amount of low-level control required from the user by allowing the prosthesis to interpret objects in its environment and assist with grasp selection and wrist positioning.

The project is mainly software- and simulation-focused. A physical Intel RealSense D435 RGB-D camera is used together with a simulated Prensilia MIA Hand in MuJoCo.

## Project Overview

The system combines computer vision, lightweight deep learning, RGB-D processing, and geometric reasoning.

The general pipeline is:

**RGB-D Camera → Object Detection/Segmentation → Object Recognition → 3D Geometry → Grasp Selection → Wrist Orientation → Simulated Prosthesis**

The camera provides both RGB and depth information. The RGB image is used for object recognition and segmentation, while the depth data is used to recover the object's 3D position and geometric properties.

Based on the perceived object, the system selects an appropriate grasp type and estimates a suitable wrist orientation. These outputs are then used to control the simulated MIA Hand.

## Main Objectives

The project focuses on six main objectives:

* Develop a lightweight vision-based object perception system for prosthetic applications.
* Compare custom CNN architectures with lightweight pretrained models and YOLO-based detectors.
* Investigate the effect of model optimization and quantization on performance.
* Evaluate whether model explanations remain consistent after optimization.
* Investigate the transfer of perception models from synthetic data to real RGB-D data.
* Integrate visual perception with RGB-D geometric reasoning for grasp selection and wrist-orientation planning.

## Custom Work

A major part of the project was developed specifically for this system rather than relying only on existing end-to-end solutions.

The custom work includes:

* Three CNN architectures:

  * Simple CNN
  * Complex CNN
  * Deeper Residual CNN
* Training and benchmarking pipelines for the different architectures.
* MobileNetV2 and MobileNetV3 comparison.
* YOLO-based object detection experiments.
* FP32, FP16 and INT8 model optimization.
* Model size, latency, memory and recognition-performance evaluation.
* Grad-CAM-based analysis of model decisions.
* Synthetic dataset generation and synthetic-to-real evaluation.
* RGB-D segmentation and depth processing.
* 3D point reconstruction from RGB-D data.
* Median-based object centroid estimation.
* PCA-based estimation of object orientation.
* Grasp-mode selection based on perceived object properties.
* Integration with the MuJoCo MIA Hand simulation.

## AI Model Evaluation

The project does not assume that a larger or more complex model is automatically better.

Instead, different approaches are compared based on their suitability for a prosthetic system, considering both recognition performance and computational requirements.

The evaluation considers:

* Accuracy
* Precision
* Recall
* F1-score
* Inference latency
* Model size
* Memory usage
* Quantization effects
* Explainability
* Synthetic-to-real performance

This makes it possible to study the trade-off between recognition quality and computational efficiency.

## RGB-D Perception

The RGB-D pipeline extends the system beyond simple object classification.

After detecting the object, depth information is used to estimate its 3D position. The system then analyses the spatial distribution of the object's points and uses Principal Component Analysis (PCA) to estimate its dominant orientation.

This information is used to determine:

* Where the object is located.
* How the object is oriented.
* Which grasp mode is appropriate.
* How the prosthetic wrist should be oriented before grasping.

## Explainable AI

Grad-CAM is used to investigate where the neural networks are focusing when making their predictions.

XAI is not included only as a visualization feature. It is used to investigate whether model optimization and quantization affect the visual reasoning of the network.

This is particularly relevant when deploying lightweight models, since a model can maintain similar classification accuracy while potentially changing the regions of the image that influence its decision.

## Simulation and Hardware

The project uses:

* **Intel RealSense D435** for RGB-D perception.
* **Prensilia MIA Hand** as the prosthetic hand platform.
* **MuJoCo** for simulation.
* **TensorFlow / Keras** for custom CNN development and training.
* **YOLO** for object detection and segmentation experiments.
* **Python** for the perception and processing pipeline.

The physical camera is used to provide real RGB-D data, while the prosthetic hand and manipulation environment are simulated.

## Overall Goal

The final goal is not to replace the prosthesis user completely.

Instead, the system is designed around **shared autonomy**: the user remains involved in the overall action, while the perception system handles repetitive and precise visual tasks such as object recognition, spatial estimation, grasp selection, and wrist positioning.

The project therefore investigates how lightweight AI and RGB-D perception can be combined to make semi-autonomous prosthetic control more practical for real-time and resource-constrained applications.

## Limitations

The current work is primarily a technical and simulation-based investigation.

It does not include:

* Clinical trials.
* Testing with people with limb loss.
* Long-term prosthesis user studies.
* Full physical integration of the perception system into a wearable prosthesis.
* Complete evaluation of real-world daily activities.

These areas remain important directions for future work.

## Author

**Endri Dibra**

M.Sc. Artificial Intelligence and Robotics
Aalborg University
