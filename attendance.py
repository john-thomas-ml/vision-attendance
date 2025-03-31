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

# Parameters
model_dir = "C:/fine_tuned_facenet.pth"
yolo_model_path = "C:/yolov8n-face.pt"
image_size = 160
num_classes = 4  # Number of students (John, Nelda, Parvathy, Safran)
GRACE_PERIOD = 30  # Time within which detection should continue (in seconds)
THRESHOLD = 300  # 300 seconds = 5 minutes to mark as present
CONSISTENCY_FRAMES = 5  # Increased to 10 for more robust tracking
COSINE_THRESHOLD = 0.45  # Tightened threshold for more accurate recognition
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLOv8 detections
MAX_TRACKING_DISTANCE = 100  # Maximum distance (in pixels) to consider two faces as the same person across frames

# Step 1: Initialize the SQLite database
def init_db():
    conn = sqlite3.connect("attendance.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            name TEXT PRIMARY KEY,
            accumulated_time INTEGER DEFAULT 0,
            last_seen DATETIME,
            marked BOOLEAN DEFAULT 0
        );
    """)
    conn.commit()
    return conn, cursor

# Step 2: Generate and save reference embeddings using the fine-tuned model
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

# Step 3: Compute cosine distance manually
def cosine_distance(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return 1 - cosine_similarity

# Step 4: Check if the person is already marked as present
def is_marked(cursor, name):
    cursor.execute("SELECT marked FROM attendance WHERE name = ?", (name,))
    result = cursor.fetchone()
    return result and result[0] == 1

# Step 5: Update the timer and mark attendance
def update_timer(cursor, conn, name, grace_period=GRACE_PERIOD, threshold=THRESHOLD):
    now = datetime.now().isoformat()
    cursor.execute("SELECT accumulated_time, last_seen FROM attendance WHERE name = ?", (name,))
    result = cursor.fetchone()

    if result:
        accumulated_time, last_seen_str = result
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

        if new_accumulated >= threshold and not is_marked(cursor, name):
            cursor.execute("UPDATE attendance SET marked = 1 WHERE name = ?", (name,))
            print(f"{name} is officially marked as PRESENT! (Total Time: {new_accumulated:.2f}s)")
        else:
            print(f"{name} is being tracked. Accumulated Time: {new_accumulated:.2f} seconds.")

    else:
        cursor.execute("INSERT INTO attendance (name, accumulated_time, last_seen) VALUES (?, 0, ?)",
                       (name, now))
        print(f"New record created for {name}. Accumulated Time: 0.00 seconds.")

    conn.commit()

# Step 6: Process live video stream with timer logic
def process_video(reference_embeddings, conn, cursor, threshold=COSINE_THRESHOLD, grace_period=GRACE_PERIOD, time_threshold=THRESHOLD, consistency_frames=CONSISTENCY_FRAMES):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load YOLOv8 face detection model
    yolo_model = YOLO(yolo_model_path)  # Use the face-specific YOLOv8 model

    # Load fine-tuned FaceNet model for recognition
    facenet_model = InceptionResnetV1(pretrained=None, classify=False, num_classes=num_classes)
    facenet_model.load_state_dict(torch.load(model_dir))
    facenet_model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dictionary to track faces across frames
    # Structure: {face_id: {"name": recognized_name, "count": consistency_count, "centroid": (x, y), "last_seen": frame_count}}
    face_tracks = {}
    face_id_counter = 0  # To assign unique IDs to detected faces
    frame_count = 0  # To track the number of frames processed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        try:
            # Use YOLOv8 for face detection
            results = yolo_model(frame, conf=YOLO_CONFIDENCE_THRESHOLD)  # Apply confidence threshold
            current_faces = []

            # Process YOLOv8 detections
            for result in results:
                for box in result.boxes:
                    # Since yolov8n-face.pt is face-specific, we don't need to check the class ID
                    x, y, w, h = map(int, box.xywh[0])  # Get bounding box coordinates (center x, y, width, height)
                    x1, y1 = x - w // 2, y - h // 2  # Convert to top-left corner
                    x2, y2 = x1 + w, y1 + h  # Bottom-right corner

                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    # Crop the face from the frame
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:  # Skip empty crops
                        continue

                    # Convert to RGB and normalize for FaceNet
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    centroid = (x, y)  # Centroid of the bounding box
                    current_faces.append({
                        "face": face_img,
                        "facial_area": {"x": x1, "y": y1, "w": w, "h": h},
                        "centroid": centroid
                    })

            print(f"Number of faces detected: {len(current_faces)}")

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
                # Batch processing for all tracked faces
                face_imgs = [transform(track["face"]) for track in face_tracks.values()]
                face_imgs = torch.stack(face_imgs)
                with torch.no_grad():
                    embeddings = facenet_model(face_imgs).numpy()

                for idx, (fid, track) in enumerate(face_tracks.items()):
                    embedding = embeddings[idx].flatten()
                    min_dist = float("inf")
                    recognized_student = None
                    distances = {}

                    # Compare with reference embeddings
                    for student, ref_embs in reference_embeddings.items():
                        for ref_emb in ref_embs:
                            dist = cosine_distance(embedding, ref_emb)
                            distances[student] = min(distances.get(student, float("inf")), dist)
                            if dist < min_dist and dist < threshold:
                                min_dist = dist
                                recognized_student = student

                    print(f"Face ID {fid} - Distances: {distances}")

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
                        if face_tracks[fid]["count"] >= consistency_frames:
                            update_timer(cursor, conn, recognized_student, grace_period, time_threshold)
                            cursor.execute("SELECT marked, accumulated_time FROM attendance WHERE name = ?", (recognized_student,))
                            status = cursor.fetchone()
                            if status:
                                marked, accumulated_time = status
                                if marked == 1:
                                    print(f"{recognized_student} is officially marked as PRESENT! (Total Time: {accumulated_time:.2f}s)")
                                else:
                                    print(f"{recognized_student} is still being tracked... (Accumulated: {accumulated_time:.2f}s)")
                    else:
                        # If the face is not recognized, mark it as "Unknown"
                        face_tracks[fid]["name"] = "Unknown"
                        face_tracks[fid]["count"] = 0
                        print(f"Face ID {fid}: No known face recognized.")

                    # Bounding box with timer label
                    x, y, w, h = track["facial_area"]["x"], track["facial_area"]["y"], track["facial_area"]["w"], track["facial_area"]["h"]
                    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
                    if w > 0 and h > 0:
                        if face_tracks[fid]["name"] != "Unknown" and face_tracks[fid]["name"] is not None:
                            cursor.execute("SELECT accumulated_time FROM attendance WHERE name = ?", (face_tracks[fid]["name"],))
                            result = cursor.fetchone()
                            accumulated_time = result[0] if result else 0
                            label = f"{face_tracks[fid]['name']} ({accumulated_time:.1f}s)"
                        else:
                            label = "Unknown"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        print("Invalid bounding box dimensions, skipping drawing.")

        except Exception as e:
            print(f"Error processing frame: {e}")

        cv2.imshow("Attendance System", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    reference_dir = "C:/dataset"
    students = ['John', 'Safran', 'Parvathy', 'Nelda']
    conn, cursor = init_db()
    reference_embeddings = generate_reference_embeddings(reference_dir, students, max_images_per_student=100)
    print("Starting live video processing...")
    process_video(reference_embeddings, conn, cursor)
    conn.close()