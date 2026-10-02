# Real-Time Helmet Detection & Traffic Violation System (YOLOv8 & Flask)

## 📌 Project Overview
This project is an AI-powered computer vision system designed to monitor motorcycle riders in real-time, detect whether they are wearing helmets, and automatically record violations. It is built using **YOLOv8** for object detection, **Flask** for the web application dashboard, and **OpenCV** for real-time video processing.

---

## 🎯 Research/Project Objectives and Methodology

**Objectives:**
1. To develop a robust object detection model capable of identifying `helmet`, `no_helmet`, `person`, and `motorcycle` in various environmental conditions.
2. To create a real-time tracking system that automatically logs violations when a rider without a helmet is detected.
3. To design a user-friendly, real-time web dashboard for authorities to monitor traffic streams and review recorded evidence.
4. To reduce manual surveillance effort and enforce traffic safety efficiently.

**Methodology:**
1. **Data Collection & Augmentation:** A dataset containing classes `helmet`, `no_helmet`, `person`, and `motorcycle` is created and annotated in YOLO format.
2. **Model Training:** YOLOv8 is trained on this dataset using Google Colab GPUs for optimal speed and accuracy. 
3. **Inference & Tracking:** OpenCV captures video frames (or images/webcam) and feeds them to the YOLO model. The model identifies objects, and a tracking algorithm ensures the same rider is not flagged multiple times simultaneously (cooldown mechanism).
4. **Backend/Dashboard:** Flask serves a real-time web interface, receiving streams, tracking statistics, and saving evidence frames to the local file system.

---

## 🏗️ Architecture and Workflow Diagram Description

The system follows a sequential pipeline:
1. **Input Layer:** Accepts Images, Pre-recorded MP4 Videos, or Live CCTV/Webcam Streams.
2. **Processing Layer (OpenCV):** Reads the media frame by frame.
3. **AI/Detection Layer (YOLOv8):** 
   - Processes the frame.
   - Outputs bounding boxes, class IDs, and confidence scores.
   - Runs object tracking (`model.track()`) to assign unique IDs to riders.
4. **Business Logic Layer:**
   - Evaluates if `class == 'no_helmet'`.
   - Checks the cooldown dictionary using the tracker ID to prevent duplicate logs.
   - If a new violation occurs, it draws a "NO HELMET" watermark, stamps the date/time, and saves the frame.
   - Increments global statistics (total riders, violations, etc.).
5. **Presentation Layer (Flask & Frontend):** 
   - Displays real-time statistics via AJAX polling.
   - Renders the live stream using MJPEG.
   - Displays a gallery of saved evidence under the "Violations" tab.

---

## 📂 Project Structure

```text
helmet_detection/
│
├── app.py                     # Main Flask web server and YOLO logic
├── requirements.txt           # Python dependencies
├── data.yaml                  # Dataset configuration for YOLO training
├── best.pt                    # Your custom-trained YOLOv8 weights (Required)
├── notebooks/
│   └── colab_training_instructions.md # Google Colab code for training
├── data/                      # Place dataset here (optional for local run)
├── uploads/                   # Temporarily stores uploaded media
├── results/                   # Stores processed output images
├── evidence/                  # Stores frames where violations are detected
├── templates/
│   ├── index.html             # Dashboard UI
│   ├── result.html            # Processed media display
│   └── violations.html        # Evidence gallery
├── static/
│   ├── css/
│   │   └── style.css          # Premium Glassmorphism styling
│   └── js/
│       └── main.js            # Real-time UI updates
└── README.md                  # Complete project documentation
```

---

## ⚙️ Installation Instructions

1. **Clone or Download the Project:**
   Extract the files into a directory (e.g., `D:\helmet detection`).

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Custom Model:**
   - Train your model using the instructions in `notebooks/colab_training_instructions.md`.
   - Download the final `best.pt` file from Colab.
   - Place `best.pt` directly inside the root folder (`helmet_detection/`).
   *(Note: If `best.pt` is missing, the app will automatically download and use `yolov8n.pt` for basic demonstration, but class mapping will be incorrect for helmet detection).*

---

## 🚀 Testing and Running Locally

1. **Start the Flask Server:**
   ```bash
   python app.py
   ```
2. **Access the Dashboard:**
   Open your web browser and go to: `http://localhost:5000`

3. **Test Features:**
   - **Upload Image/Video:** Drag and drop an image or video to see detection bounding boxes.
   - **Live Camera:** Click "Start Live Stream" to activate your webcam.
   - **Check Violations:** Navigate to the "Violations" page from the sidebar to see saved snapshots of riders without helmets.

---

## 🧠 Viva Preparation (Q&A)

**Q1: Why did you choose YOLOv8 over other algorithms like Faster R-CNN?**
*A1:* YOLOv8 is a single-stage detector, making it incredibly fast and suitable for real-time applications like traffic monitoring, whereas Faster R-CNN (two-stage) is highly accurate but too slow for real-time high-FPS video streams.

**Q2: How do you prevent the system from saving 100 images of the same rider without a helmet?**
*A2:* I utilized YOLO's built-in tracking capabilities (`model.track()`) combined with a custom time-based cooldown mechanism. When a violation is detected for a specific Track ID, the system records it and ignores subsequent frames of that exact ID for a set duration (e.g., 10 seconds).

**Q3: Explain the `data.yaml` file.**
*A3:* The `data.yaml` file tells the YOLO model where to find the training and validation images, and defines the number of classes (`nc`) and their names (`helmet, no_helmet, person, motorcycle`).

**Q4: What metrics define a "good" model during training?**
*A4:* We look at **mAP50** (mean Average Precision at 0.5 IoU) and **mAP50-95**. High mAP values combined with decreasing training/validation loss graphs indicate the model is learning successfully without overfitting.

---

## 🔮 Future Enhancements
1. **License Plate Recognition (ANPR):** Integrate an OCR model (like Tesseract or EasyOCR) to automatically read the number plate of the violating motorcycle.
2. **Database Integration:** Connect the Flask app to PostgreSQL or MongoDB to store violation logs persistently rather than relying on global memory variables.
3. **Email/SMS Alerts:** Integrate Twilio or SMTP to immediately alert authorities when a violation occurs.
4. **Night Vision Optimization:** Train the model with IR/Night-vision datasets for 24/7 reliability.
