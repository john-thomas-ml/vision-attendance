from flask import Flask, render_template, Response, jsonify, request
import cv2
import sqlite3
from datetime import datetime
import os
import pickle
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import threading
import time
import json

# Parameters
model_dir = "fine_tuned_arcface.pth"
yolo_model_path = "yolov8n-face.pt"
image_size = 160
num_classes = 4  # Number of students (John, Nelda, Parvathy, Safran)
GRACE_PERIOD = 30  # Time within which detection should continue (in seconds)
THRESHOLD = 300  # 300 seconds = 5 minutes to mark as present
CONSISTENCY_FRAMES = 5  # Number of frames for consistent recognition
COSINE_THRESHOLD = 0.45  # Threshold for face recognition
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLOv8 detections
MAX_TRACKING_DISTANCE = 100  # Maximum distance (in pixels) to track faces
REFERENCE_DIR = "dataset"
STUDENTS = ['John', 'Safran', 'Parvathy', 'Nelda']

app = Flask(__name__)

# Global variables
camera = None
is_streaming = False
stream_lock = threading.Lock()
attendance_data = {}
face_tracks = {}
face_id_counter = 0
frame_count = 0

# Database connection
def get_db_connection():
    conn = sqlite3.connect("attendance.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            name TEXT PRIMARY KEY,
            accumulated_time INTEGER DEFAULT 0,
            last_seen DATETIME,
            marked BOOLEAN DEFAULT 0
        );
    """)
    
    # Initialize records for all students if they don't exist
    for student in STUDENTS:
        cursor.execute("INSERT OR IGNORE INTO attendance (name, accumulated_time, last_seen, marked) VALUES (?, 0, NULL, 0)",
                     (student,))
    
    conn.commit()
    conn.close()

# Generate reference embeddings
def generate_reference_embeddings(reference_dir, students, embedding_file="reference_embeddings.pkl", max_images_per_student=50):
    if os.path.exists(embedding_file):
        print(f"Loading existing embeddings from {embedding_file}")
        with open(embedding_file, "rb") as f:
            return pickle.load(f)

    print("Generating reference embeddings...")
    model = InceptionResnetV1(pretrained=None, classify=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    reference_embeddings = {}
    for student in students:
        student_dir = os.path.join(reference_dir, student)
        if not os.path.exists(student_dir):
            print(f"Directory {student_dir} does not exist. Skipping {student}.")
            continue

        valid_extensions = ('.jpg', '.png')
        image_files = [f for f in os.listdir(student_dir) if f.lower().endswith(valid_extensions)]
        image_files = image_files[:max_images_per_student]
        total_images = len(image_files)
        print(f"Processing {total_images} images for {student}...")

        if total_images == 0:
            print(f"No valid images (.jpg or .png) found in {student_dir}")

        embeddings = []
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(student_dir, img_file)
            print(f"Processing {img_path} ({i+1}/{total_images})")
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    embedding = model(img).numpy().flatten()
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if embeddings:
            reference_embeddings[student] = embeddings
            print(f"Generated {len(embeddings)} embeddings for {student}")
        else:
            print(f"No embeddings generated for {student}")

    with open(embedding_file, "wb") as f:
        pickle.dump(reference_embeddings, f)
    print(f"Saved embeddings to {embedding_file}")
    return reference_embeddings

# Compute cosine distance
def cosine_distance(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return 1 - cosine_similarity

# Load models
def load_models():
    # Load YOLOv8 face detection model
    yolo_model = YOLO(yolo_model_path)
    
    # Load fine-tuned ArcFace model for recognition
    arcface_model = InceptionResnetV1(pretrained=None, classify=False, num_classes=num_classes)
    arcface_model.load_state_dict(torch.load(model_dir))
    arcface_model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return yolo_model, arcface_model, transform

ATTENDANCE_THRESHOLD = THRESHOLD  # Set to default value

# Add this new route
@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    global ATTENDANCE_THRESHOLD
    data = request.get_json()
    threshold = data.get('threshold', THRESHOLD)
    
    # Update the threshold
    ATTENDANCE_THRESHOLD = threshold
    
    # Update all student records with the new threshold
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, accumulated_time FROM attendance")
    students = cursor.fetchall()
    
    for student in students:
        name = student['name']
        time = student['accumulated_time']
        # Update student status based on new threshold
        marked = 1 if time >= ATTENDANCE_THRESHOLD else 0
        cursor.execute("UPDATE attendance SET marked = ? WHERE name = ?", (marked, name))
    
    conn.commit()
    conn.close()
    
    return jsonify({'status': 'success', 'threshold': threshold})

# Update attendance timer
def update_timer(name, grace_period=GRACE_PERIOD, threshold=None):
    if threshold is None:
        threshold = ATTENDANCE_THRESHOLD  # Use the global threshold
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    cursor.execute("SELECT accumulated_time, last_seen, marked FROM attendance WHERE name = ?", (name,))
    result = cursor.fetchone()

    if result:
        accumulated_time, last_seen_str, marked = result
        last_seen = datetime.fromisoformat(last_seen_str) if last_seen_str else None

        if last_seen:
            time_since_last_seen = (datetime.now() - last_seen).total_seconds()
        else:
            time_since_last_seen = 0

        if 0 < time_since_last_seen <= grace_period:
            new_accumulated = accumulated_time + time_since_last_seen
        else:
            new_accumulated = accumulated_time

        cursor.execute("UPDATE attendance SET accumulated_time = ?, last_seen = ? WHERE name = ?",
                       (new_accumulated, now, name))

        if new_accumulated >= threshold and marked == 0:
            cursor.execute("UPDATE attendance SET marked = 1 WHERE name = ?", (name,))
            print(f"{name} is officially marked as PRESENT! (Total Time: {new_accumulated:.2f}s)")

    conn.commit()
    conn.close()

# Get current attendance data
def get_attendance_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, accumulated_time, marked FROM attendance")
    results = cursor.fetchall()
    
    data = {}
    for row in results:
        data[row['name']] = {
            'time': row['accumulated_time'],
            'status': 'Present' if row['marked'] == 1 else 'Absent'
        }
    
    conn.close()
    return data

# Clear attendance records
def clear_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE attendance SET accumulated_time = 0, last_seen = NULL, marked = 0")
    conn.commit()
    conn.close()
    
    # Update global attendance data
    global attendance_data
    attendance_data = get_attendance_data()

# Process video stream
def process_frame(frame, yolo_model, arcface_model, transform, reference_embeddings):
    global face_tracks, face_id_counter, frame_count
    
    try:
        # Use YOLOv8 for face detection
        results = yolo_model(frame, conf=YOLO_CONFIDENCE_THRESHOLD)
        current_faces = []

        # Process YOLOv8 detections
        for result in results:
            for box in result.boxes:
                x, y, w, h = map(int, box.xywh[0])
                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x1 + w, y1 + h

                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Crop the face from the frame
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:  # Skip empty crops
                    continue

                # Convert to RGB for recognition
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                centroid = (x, y)
                current_faces.append({
                    "face": face_img,
                    "facial_area": {"x": x1, "y": y1, "w": w, "h": h},
                    "centroid": centroid
                })

        # Match current faces to existing tracks using centroid distance
        new_face_tracks = {}
        matched_ids = set()

        for face in current_faces:
            centroid = face["centroid"]
            min_dist = float("inf")
            matched_id = None

            # Find the closest existing track
            for fid, track in face_tracks.items():
                track_centroid = track["centroid"]
                dist = np.sqrt((centroid[0] - track_centroid[0])**2 + (centroid[1] - track_centroid[1])**2)
                if dist < min_dist and dist < MAX_TRACKING_DISTANCE and fid not in matched_ids:
                    min_dist = dist
                    matched_id = fid

            if matched_id is not None:
                # Update existing track
                new_face_tracks[matched_id] = face_tracks[matched_id]
                new_face_tracks[matched_id]["centroid"] = centroid
                new_face_tracks[matched_id]["last_seen"] = frame_count
                new_face_tracks[matched_id]["face"] = face["face"]
                new_face_tracks[matched_id]["facial_area"] = face["facial_area"]
                matched_ids.add(matched_id)
            else:
                # Create a new track
                new_face_tracks[face_id_counter] = {
                    "name": None,
                    "count": 0,
                    "centroid": centroid,
                    "last_seen": frame_count,
                    "face": face["face"],
                    "facial_area": face["facial_area"]
                }
                matched_ids.add(face_id_counter)
                face_id_counter += 1

        # Update face tracks and remove old tracks (not seen for 30 frames)
        face_tracks = {fid: track for fid, track in new_face_tracks.items() if frame_count - track["last_seen"] < 30}

        # Process each tracked face for recognition
        if face_tracks:
            face_imgs = []
            face_ids = []
            
            for fid, track in face_tracks.items():
                face_imgs.append(transform(track["face"]))
                face_ids.append(fid)
                
            if face_imgs:
                face_imgs = torch.stack(face_imgs)
                with torch.no_grad():
                    embeddings = arcface_model(face_imgs).numpy()

                for idx, fid in enumerate(face_ids):
                    embedding = embeddings[idx].flatten()
                    min_dist = float("inf")
                    recognized_student = None

                    # Compare with reference embeddings
                    for student, ref_embs in reference_embeddings.items():
                        for ref_emb in ref_embs:
                            dist = cosine_distance(embedding, ref_emb)
                            if dist < min_dist and dist < COSINE_THRESHOLD:
                                min_dist = dist
                                recognized_student = student

                    # Handle recognition result
                    if recognized_student:
                        # If the face is recognized, update its track
                        if face_tracks[fid]["name"] == recognized_student:
                            face_tracks[fid]["count"] += 1
                        else:
                            # If the name has changed, reset the count
                            face_tracks[fid]["name"] = recognized_student
                            face_tracks[fid]["count"] = 1

                        # If the consistency threshold is met, update the timer
                        if face_tracks[fid]["count"] >= CONSISTENCY_FRAMES:
                            update_timer(recognized_student)
                    else:
                        # If the face is not recognized, mark it as "Unknown"
                        face_tracks[fid]["name"] = "Unknown"
                        face_tracks[fid]["count"] = 0

                    # Draw bounding box and label
                    x, y, w, h = face_tracks[fid]["facial_area"]["x"], face_tracks[fid]["facial_area"]["y"], face_tracks[fid]["facial_area"]["w"], face_tracks[fid]["facial_area"]["h"]
                    if w > 0 and h > 0:
                        if face_tracks[fid]["name"] != "Unknown" and face_tracks[fid]["name"] is not None:
                            label = face_tracks[fid]["name"]
                            color = (0, 255, 0)  # Green for recognized faces
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)  # Red for unknown faces
                            
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        frame_count += 1
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        
    return frame

# Video streaming generator function
def gen_frames():
    global camera, is_streaming
    
    # Load models
    yolo_model, arcface_model, transform = load_models()
    
    # Load reference embeddings
    reference_embeddings = generate_reference_embeddings(REFERENCE_DIR, STUDENTS)
    
    while is_streaming:
        success, frame = camera.read()
        if not success:
            break
        
        # Process the frame
        processed_frame = process_frame(frame, yolo_model, arcface_model, transform, reference_embeddings)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Update attendance data for the frontend
        global attendance_data
        attendance_data = get_attendance_data()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Start video stream
def start_camera():
    global camera, is_streaming
    with stream_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Error: Could not open webcam.")
                return False
        is_streaming = True
    return True

# Stop video stream
def stop_camera():
    global camera, is_streaming
    with stream_lock:
        is_streaming = False
        if camera is not None:
            camera.release()
            camera = None
    return True

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    if start_camera():
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Failed to start camera'})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    if stop_camera():
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Failed to stop camera'})

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    return jsonify(attendance_data)

@app.route('/clear_attendance', methods=['POST'])
def clear_attendance_route():
    clear_attendance()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Initialize the database
    init_db()
    
    # Start the Flask app
    app.run(debug=True, port=3001)