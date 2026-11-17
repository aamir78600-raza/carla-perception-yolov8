# carla-perception-yolov8
Real-time object detection for autonomous driving in CARLA using fine-tuned YOLOv8, with a Gradio demo. Includes a Perceptron → MLP → CNN comparison.


# 🚗 CARLA Perception: Object Detection for Autonomous Driving

A perception module for autonomous driving built on the CARLA simulator. The project compares classic neural network architectures (Perceptron, MLP, CNN) and fine-tunes a pre-trained **YOLOv8n** model to detect vehicles, pedestrians, cyclists, traffic lights and traffic signs, with an interactive **Gradio** web demo.

## 📌 Overview

| Item | Details |
|------|---------|
| Task | Real-time object detection for autonomous driving |
| Model | YOLOv8n (transfer learning from pre-trained weights) |
| Dataset | CARLA dataset from Roboflow Universe (v20, CC BY 4.0) |
| Framework | PyTorch, Ultralytics |
| Training hardware | NVIDIA Tesla T4 (Google Colab) |
| Demo | Gradio web app |

## 🎯 Detected Classes (10)

`bike` · `motobike` · `person` · `traffic_light_green` · `traffic_light_orange` · `traffic_light_red` · `traffic_sign_30` · `traffic_sign_60` · `traffic_sign_90` · `vehicle`

## 📂 Dataset

- **Source:** [CARLA dataset on Roboflow Universe](https://universe.roboflow.com/alec-hantson-student-howest-be/carla-izloa/dataset/20) (version 20)
- **Training images:** 1,600
- **Validation images:** 264
- **Format:** YOLOv8
- **License:** CC BY 4.0

## 🧠 Project Structure

The notebook is organised into the following sections:

1. **Setup & Dataset:** environment setup, dataset download and exploration
2. **Single Layer Perceptron:** baseline model; limited to linear patterns
3. **Multi Layer Perceptron:** non-linear learning, but loses spatial information
4. **Convolutional Neural Network:** conv + batch norm + pooling blocks for spatial feature extraction
5. **Transfer Learning with YOLOv8:** fine-tuning a pre-trained detector on CARLA data
6. **Evaluation & Demo:** metrics, visualisation and Gradio interface

## 📊 Results (YOLOv8n, validation set)

| Metric | Score |
|--------|-------|
| mAP@50 | **0.896** |
| mAP@50-95 | **0.641** |
| Precision | 0.786 |
| Recall | 0.841 |

Trained for 5 epochs at 640×640 with the Adam optimizer (lr = 0.001), batch size 16.

**Strongest classes:** person, traffic_light_red, traffic_light_orange (mAP@50 > 0.97).
**Weakest classes:** traffic_sign_30 and traffic_sign_60 (mAP@50 ≈ 0.71–0.73), which are visually similar and small in the frame.

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/<your-username>/carla-perception-yolov8.git
cd carla-perception-yolov8
pip install ultralytics torch torchvision opencv-python matplotlib numpy roboflow gradio
```

### Download the dataset
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("alecs-place").project("carla-izloa")
dataset = project.version(20).download("yolov8")
```

### Train
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="CARLA-20/data.yaml", epochs=5, imgsz=640, batch=16,
            optimizer="Adam", lr0=0.001, name="carla_perception")
```

### Run the demo
```python
from ultralytics import YOLO
model = YOLO("models/carla_perception/weights/best.pt")
results = model("path/to/image.jpg")
results[0].show()
```
The notebook also launches a Gradio app where you can upload an image and view detections.

## 🔮 Future Work

- Train for more epochs and use a larger model (YOLOv8s/m)
- Add more samples of traffic signs to improve weak classes
- Integrate with a live CARLA simulation feed for real-time inference
- Add a test split for final evaluation

## 🙏 Acknowledgements

- [CARLA Simulator](https://carla.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Roboflow Universe dataset by Alec Hantson (Howest)

## 📄 License

MIT License (code) · Dataset under CC BY 4.0
