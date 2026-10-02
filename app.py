import os
import cv2
import time
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
EVIDENCE_FOLDER = 'evidence'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['EVIDENCE_FOLDER'] = EVIDENCE_FOLDER

# --- Load YOLO Model ---
MODEL_PATH = 'best.pt'
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print("Custom model loaded successfully.")
else:
    print(f"WARNING: {MODEL_PATH} not found. Loading pre-trained YOLOv8n for demonstration.")
    model = YOLO('yolov8n.pt') # Will download if missing

# --- Initialize OCR ---
print("Initializing EasyOCR (this may take a moment)...")
reader = easyocr.Reader(['en'], gpu=False)

# Classes expected: 0: helmet, 1: no_helmet, 2: person, 3: motorcycle
# If using standard yolov8n, classes won't match. We handle it generically but assume custom model for logic.

# --- Global Tracking & Statistics ---
stats = {
    'total_riders': 0,
    'helmet_users': 0,
    'no_helmet_users': 0,
    'total_violations': 0
}

# Cooldown dictionary for tracking IDs to avoid saving the same violation multiple times
# Structure: { track_id: last_violation_timestamp }
violation_cooldowns = {}
COOLDOWN_SECONDS = 10 

def process_frame(frame, run_tracking=False):
    """
    Process a single frame using YOLO, draw bounding boxes, and handle violations.
    """
    global stats, violation_cooldowns
    
    # Run YOLO inference
    # Use track if run_tracking is True to get object IDs
    if run_tracking:
        results = model.track(frame, persist=True, verbose=False, conf=0.5)
    else:
        results = model(frame, verbose=False, conf=0.5)
        
    annotated_frame = frame.copy()
    current_time = time.time()
    
    # Check if there are any detections
    if results and len(results) > 0 and results[0].boxes:
        boxes = results[0].boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            # Check if model has custom class names, else use defaults
            class_name = model.names.get(cls_id, str(cls_id))
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            
            # --- SIMULATION HACK FOR DEMONSTRATION ---
            # If using standard yolov8n, we map 'person' to 'no_helmet' and 'car'/'motorcycle' to 'helmet' 
            # so the dashboard actually does something when you test it on standard objects.
            if not os.path.exists(MODEL_PATH):
                if class_name == 'person':
                    class_name = 'no_helmet'
                elif class_name in ['car', 'motorcycle', 'truck']:
                    class_name = 'helmet'
                elif class_name == 'bicycle':
                    class_name = 'person'
                    
            # --- Visualization ---
            color = (0, 255, 0) # Green for helmet / safe
            if class_name in ['no_helmet', 'Without Helmet']:
                color = (0, 0, 255) # Red for violation
            elif class_name in ['person', 'motorcycle']:
                color = (255, 0, 0) # Blue for rider/bike
                
            label = f"{class_name} {conf:.2f}"
            if track_id != -1:
                label += f" ID:{track_id}"
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # --- Violation Logic ---
            if class_name in ['no_helmet', 'Without Helmet']:
                # Determine if we should record a violation based on cooldown
                record_violation = False
                
                if track_id != -1:
                    # Using tracking
                    if track_id not in violation_cooldowns or (current_time - violation_cooldowns[track_id] > COOLDOWN_SECONDS):
                        violation_cooldowns[track_id] = current_time
                        record_violation = True
                else:
                    # No tracking (e.g., single image), just record it
                    record_violation = True
                    
                if record_violation:
                    stats['no_helmet_users'] += 1
                    stats['total_violations'] += 1
                    
                    # Save evidence frame
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    evidence_filename = f"violation_{timestamp_str}.jpg"
                    evidence_path = os.path.join(app.config['EVIDENCE_FOLDER'], evidence_filename)
                    
                    # --- OCR License Plate Detection ---
                    plate_text = "UNKNOWN"
                    try:
                        # Focus OCR on the bottom 30% of the bounding box where the plate is
                        h = y2 - y1
                        y_start = int(y1 + (h * 0.70))
                        violator_crop = frame[max(0, y_start):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        
                        ocr_results = reader.readtext(violator_crop)
                        texts = [res[1] for res in ocr_results if len(res[1]) >= 2] # Filter tiny noise
                        if texts:
                            plate_text = " ".join(texts).upper()
                    except Exception as e:
                        print("OCR Error:", e)
                    
                    # Add 'NO HELMET' watermark
                    cv2.putText(annotated_frame, "VIOLATION: NO HELMET", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(annotated_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if plate_text != "UNKNOWN":
                        cv2.putText(annotated_frame, f"PLATE: {plate_text}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        print(f"Violation Logged! Plate: {plate_text}")
                    
                    cv2.imwrite(evidence_path, annotated_frame)
                    
            elif class_name in ['helmet', 'With Helmet']:
                # Just count stats, maybe with cooldown if tracking
                if track_id != -1:
                    pass # Advanced: could track unique helmet users
                else:
                    stats['helmet_users'] += 1
                    
            elif class_name == 'person':
                stats['total_riders'] += 1

    return annotated_frame

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
        
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Check if image or video
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ['jpg', 'jpeg', 'png']:
            # Process image
            img = cv2.imread(file_path)
            processed_img = process_frame(img, run_tracking=False)
            
            res_filename = f"res_{filename}"
            res_path = os.path.join(app.config['RESULT_FOLDER'], res_filename)
            cv2.imwrite(res_path, processed_img)
            
            return render_template('result.html', media_type='image', media_url=url_for('static', filename=f'../results/{res_filename}'))
            
        elif ext in ['mp4', 'avi', 'mov']:
            # For video, we will redirect to a video processing page
            return render_template('result.html', media_type='video', video_name=filename)
            
    return redirect(url_for('index'))

def generate_video_stream(video_path=None):
    if video_path is None:
        # Try multiple camera indices if the default (0) fails
        cap = None
        for i in range(3):
            # Try DirectShow first (often fixes issues on Windows)
            temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not temp_cap.isOpened():
                temp_cap = cv2.VideoCapture(i)
                
            if temp_cap.isOpened():
                cap = temp_cap
                print(f"Successfully opened camera at index {i}", flush=True)
                break
        
        if cap is None or not cap.isOpened():
            print("ERROR: Could not open any camera (indices 0, 1, 2).", flush=True)
            # We yield a blank image or just break to stop the feed
            return
    else:
        cap = cv2.VideoCapture(video_path)
        
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        processed_frame = process_frame(frame, run_tracking=True)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/video_feed')
def video_feed():
    video_name = request.args.get('video_name')
    if video_name:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
        return Response(generate_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Default to webcam
        return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violations')
def violations():
    # List all evidence images
    evidence_files = []
    if os.path.exists(EVIDENCE_FOLDER):
        files = os.listdir(EVIDENCE_FOLDER)
        files.sort(reverse=True) # Newest first
        evidence_files = files
    return render_template('violations.html', violations=evidence_files)

@app.route('/evidence/<filename>')
def get_evidence(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['EVIDENCE_FOLDER'], filename)

@app.route('/results/<filename>')
def get_result(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/stats')
def get_stats():
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
