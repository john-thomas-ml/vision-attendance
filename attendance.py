import sqlite3
import cv2
import os
from deepface import DeepFace
from datetime import datetime

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

GRACE_PERIOD = 5  # Time within which detection should continue (in seconds)
THRESHOLD = 10  # 600 seconds = 10 minutes

cap = cv2.VideoCapture(0) 

def extract_name(path):
    parts = os.path.normpath(path).split(os.path.sep)
    return parts[-2] if len(parts) > 1 else "Unknown"


def is_marked(name):
    """Check if the person is already marked as present."""
    cursor.execute("SELECT marked FROM attendance WHERE name = ?", (name,))
    result = cursor.fetchone()
    return result and result[0] == 1

def update_timer(name):
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

        if 0 < time_since_last_seen <= GRACE_PERIOD:
            new_accumulated = accumulated_time + time_since_last_seen
        else:
            new_accumulated = accumulated_time 

        cursor.execute("UPDATE attendance SET accumulated_time = ?, last_seen = ? WHERE name = ?",
                       (new_accumulated, now, name))

        if new_accumulated >= THRESHOLD and not is_marked(name):
            cursor.execute("UPDATE attendance SET marked = 1 WHERE name = ?", (name,))
            print(f"✅ {name} is officially marked as PRESENT! (Total Time: {new_accumulated:.2f}s)")
        else:
            print(f"⏳ {name} is being tracked. Accumulated Time: {new_accumulated:.2f} seconds.")

    else:
        cursor.execute("INSERT INTO attendance (name, accumulated_time, last_seen) VALUES (?, 0, ?)",
                       (name, now))

    conn.commit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame.")
        break
    
    try:
        cv2.imwrite("temp.jpg", frame) 
        result = DeepFace.find(img_path="temp.jpg", db_path="dataset/", model_name="Facenet", detector_backend="opencv")

        if isinstance(result, list) and len(result) > 0:
            for res in result:
                if not res.empty:
                    name = extract_name(res['identity'][0])
                    print(f"🔍 Recognized: {name}")
                    update_timer(name)

                    cursor.execute("SELECT marked, accumulated_time FROM attendance WHERE name = ?", (name,))
                    status = cursor.fetchone()
                    if status:
                        marked, accumulated_time = status
                        if marked == 1:
                            print(f"✅ {name} is officially marked as PRESENT! (Total Time: {accumulated_time:.2f}s)")
                        else:
                            print(f"⏳ {name} is still being tracked... (Accumulated: {accumulated_time:.2f}s)")
                    else:
                        print(f"⚠️ No database record found for {name}")
        else:
            print("⚠️ No known face recognized.")

    except Exception as e:
        print(f"❌ Error in face recognition: {str(e)}")
    
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
