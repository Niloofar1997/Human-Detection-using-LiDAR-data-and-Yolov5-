# Human-Detection-using-LiDAR-data-and-Yolov5-
A transfer learning approach

## Introduction

**Project Overview:**
This project focuses on enhancing continuous patient monitoring within healthcare systems—a crucial element for ensuring patient safety and well-being. Traditional monitoring methods, typically camera-based, require extensive oversight and can intrude on patient privacy. To address these challenges, LiDAR technology is utilized as a superior alternative, offering effective monitoring across diverse lighting conditions without compromising privacy. LiDAR technology distinguishes itself by providing high-resolution depth data that respects patient confidentiality, making it an optimal choice for environments where privacy is paramount.

**Technology and Approach:**
LiDAR is chosen for its **precision** and its ability to deliver detailed **depth information** critical for accurate patient monitoring. Utilizing a digital LiDAR sensor, we developed a system that employs a YOLOv5 deep learning model, enhanced through transfer learning, to accurately detect and track human presence within a room in real-time. This setup ensures precise tracking capabilities essential for critical healthcare interventions, enhancing response efficiency and safety during high-risk periods such as disease outbreaks.

**Objectives and Outcomes:**
The primary objective of this initiative is to seamlessly integrate our LiDAR-based monitoring solution into Ambient Assisted Living (AAL) environments, particularly to aid the elderly in maintaining a safe and autonomous lifestyle. The performance of our YOLOv5 model has been noteworthy, achieving a Mean Average Precision (mAP) of 99.4% with an Intersection over Union (IoU) threshold of 0.5, complemented by a Precision of 99.6% and a Recall of 99.4%.

## Installation and Setup Instructions

Installation of Ouster Python SDK is necessary, as we are using LiDAR data comming from Ouster's LiDAR sensor. the installation instructions can be found [here](https://static.ouster.dev/sdk-docs/).
To begin using the YOLOv5 model with LiDAR technology, please follow these steps:

1. Clone the Repository Start by cloning the official YOLOv5 repository from Ultralytics. Open your command line interface and enter the following command:
```bash
git clone https://github.com/ultralytics/yolov5
```

2. Change to the Repository Directory After the cloning process is complete, navigate into the 'yolov5' directory to make sure all further commands are executed from the correct location:

```bash
cd yolov5
```

3.  Install the requirements
```bash
pip install -r requirements.txt
```
4. The `detect.py` script, located in the source folder of the YOLOv5 repository, is capable of performing inference on diverse input types such as images, videos, live video feeds, and webcams. For instance, if we aim to identify individuals in an image using the pre-trained YOLOv5s model with a confidence threshold set at 40%, the required command for execution in a terminal within the source directory is as follows:
```bash
python detect.py --class 0 --weights Yolov5s.pt --conf-thres=0.4 --source example_pic.jpeg --view-img
```
This automatically stores the outcomes in the folder runs/detect/exp as labeled images with annotations displaying the confidence levels of the predictions.

**Extra Information**

YOLOv5, designed for 2D images, contrasts with the 3D nature of LiDAR data. To adapt YOLOv5, 3D LiDAR data is converted to 2D by extracting the reflectivity layer, making it suitable for the model.
