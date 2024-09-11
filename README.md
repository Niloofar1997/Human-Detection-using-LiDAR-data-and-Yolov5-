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

*Installation of Ouster Python SDK is necessary, as we are using LiDAR data comming from Ouster's LiDAR sensor. the installation instructions can be found [here](https://static.ouster.dev/sdk-docs/).*

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

## Inference with Object Detection 
To run the  trained model on (pre-recorded) LiDAR data, run the following command:
```bash

### *Extra Information* 

YOLOv5, designed for 2D images, contrasts with the 3D nature of LiDAR data. To adapt YOLOv5, 3D LiDAR data is converted to 2D by extracting the reflectivity layer, making it suitable for the model.
YOLOv5 processes images but not PCAP files from Ouster LiDAR sensors. To use Ouster data, it must be converted into image formats, aided by the Ouster Python SDK. This SDK adjusts the `detect.py` file in the YOLOv5 repository to analyze the reflectivity layer from PCAP files, creating a customized script detailed later.
The Ouster sensor produces PCAP files, containing raw UDP packets, and JSON files with crucial metadata, interpreted using the SDK’s client module. Installation of the Ouster Python SDK is necessary.

Dataset preparation involved recordings with the OSDome sensor in various positions and times to capture different spatial perspectives and lighting conditions, enhancing dataset diversity and robustness. Additional data from an OS0 sensor on Ouster’s website further diversified the dataset.

The collection resulted in PCAP and JSON file pairs, moving the project to the image extraction stage.
**Image Extraction:** Using the Ouster client and pcap modules, we read metadata from JSON files and point cloud data from PCAP files. The script iterates through scans from the PCAP, extracts the reflectivity data, and applies the client's destagger function to correct pixel staggering in raw LiDAR data, improving visual clarity. Each corrected scan is then converted to an 8-bit JPEG image, forming a sequential dataset.
```bash
with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())
source = pcap.Pcap(pcap_path, metadata)
with closing(client.Scans(source)) as scans:
    for scan in scans:
        ref_field = scan.field(client.ChanField.REFLECTIVITY)
        ref_val = client.destagger(source.metadata, ref_field)
        ref_img = ref_val.astype(np.uint8)
        filename = 'extract'+str(counter)+'.jpg'
        cv2.imwrite(img_path+filename, ref_img)
```
**Preprocessing Phase:** Images are loaded in grayscale and prepared through standardization and noise reduction. Each image is resized to uniform dimensions, histogram equalization is applied to enhance feature visibility, and wavelet denoising is used to reduce noise while preserving details, readying the dataset for labeling.
```bash
for file in image_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)
    canvas[y_off:y_off+height, x_off:x_off+width] = img
    img_resize = cv2.resize(canvas, (640, 640))
    img_eq = cv2.equalizeHist(img_resize)
    coeffs = pywt.dwt2(img_eq, 'haar')
    rec = pywt.idwt2(coeffs, 'haar')
    rec = np.uint8(rec)
    cv2.imwrite(result_filename, rec)
```
In the next phase, we focused on training a custom YOLOv5 model using a Python script in a GPU-enabled Google Colab environment. The script was tailored with specific commands to optimize learning:
```bash
!python train.py --img 640 --single-cls --batch 16 --epochs 30 --data /content/datasets/human-detection-1/data.yaml --weights yolov5s.pt
```
As a transfer learning approach, the **yolov5s** weight was used 

Additionally, the [Roboflow](https://roboflow.com/) Python package was used to manage dataset preparation:
```bash
%pip install -q roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="API_KEY_HERE")
project = rf.workspace("hochschule-wismar-zsiui").project("human-detection-o2d8k")
dataset = project.version(1).download("yolov5")
```
**Data Reception & Image Conversion:**

The pipeline begins by checking if the incoming data is in PCAP format using if `is_pcap`. After identifying the data format, it extracts sensor metadata from JSON files for processing:
```bash
with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())
```
A video writer is then set up to save processed data as video files:
```bash
vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
```
LiDAR data is processed to convert scans into a pseudo-RGB format for YOLOv5 compatibility:
```bash
for scan in scans:
    ref_field = scan.field(client.ChanField.REFLECTIVITY)
    ref_val = client.destagger(pcap_file.metadata, ref_field)
    combined_img = np.dstack((ref_val, ref_val, ref_val))
```
The processed data is prepared for YOLOv5 model inference:
```bash
dataset = LoadNumpy(numpy=combined_img, path="", img_size=imgsz, stride=stride, auto=pt and not jit)
if pt and device.type != 'cpu':
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))
```
Images are processed and predictions are refined using Non-Maximum Suppression (NMS) to ensure accuracy:
```bash
for path, im, im0s, vid_cap, s in dataset:
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
```
Detected objects are annotated and the results are displayed:
```bash
for i, det in enumerate(pred):
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
```
