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

To begin using the YOLOv5 model for patient monitoring with LiDAR technology, please follow these straightforward steps:

### 1. Clone the Repository Start by cloning the official YOLOv5 repository from Ultralytics. Open your command line interface and enter the following command:
```bash
git clone https://github.com/ultralytics/yolov5```

### 2. Change to the Repository Directory After the cloning process is complete, navigate into the 'yolov5' directory to make sure all further commands are executed from the correct location:

```bash
cd yolov5```
