# Real-Time Helmet Detection - YOLO Training Instructions (Google Colab)

## 1. Setup and Preparation
First, open Google Colab, create a new notebook, and change the runtime type to GPU (Runtime -> Change runtime type -> Hardware accelerator -> GPU).

### Cell 1: Check GPU Availability
```python
!nvidia-smi
```

### Cell 2: Install Ultralytics
```python
%pip install ultralytics
import ultralytics
ultralytics.checks()
```

### Cell 3: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Dataset Preparation
Upload your dataset to Google Drive in YOLO format (images and labels folders).
Create a `data.yaml` file in the dataset directory. 
Verify the structure in Colab:
```python
import os
dataset_path = '/content/drive/MyDrive/helmet_dataset'
print("Files in dataset path:", os.listdir(dataset_path))
```

## 3. Training the Model
Use YOLOv8 or YOLO11 to train the custom dataset. 
Adjust parameters based on your requirements:
- `epochs`: Number of training rounds (e.g., 50-100).
- `imgsz`: Image size (e.g., 640).
- `batch`: Batch size, dependent on GPU memory (e.g., 16 or 32).
- `lr0`: Initial learning rate (default is 0.01).

### Cell 4: Train YOLO Model
```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano size for speed)
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='/content/drive/MyDrive/helmet_dataset/data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=16,
    lr0=0.01,
    name='helmet_detection_model'
)
```

## 4. Validation and Metrics
The training process automatically validates at the end of each epoch.
To explicitly validate the best model on the validation set:

### Cell 5: Validate Best Model
```python
# Load the best model weights
best_model = YOLO('runs/detect/helmet_detection_model/weights/best.pt')

# Validate on validation set
metrics = best_model.val()
print("mAP50-95:", metrics.box.map)    # map50-95
print("mAP50:", metrics.box.map50)  # map50
```

## 5. Testing and Inference
Test the trained model on an image or video to see predictions.

### Cell 6: Run Inference on Image
```python
results = best_model.predict(source='path/to/test/image.jpg', save=True, show=False)
```

### Cell 7: Display Result
```python
import matplotlib.pyplot as plt
import cv2

# Read the saved image with detections
img_path = results[0].save_dir + '/' + os.path.basename('path/to/test/image.jpg')
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## 6. Save the Model
The best weights are saved in `runs/detect/helmet_detection_model/weights/best.pt`.
Download this file and place it in the root directory of your Flask application for deployment.
```python
import shutil
shutil.copy('runs/detect/helmet_detection_model/weights/best.pt', '/content/drive/MyDrive/best.pt')
print("Model saved to Google Drive!")
```
