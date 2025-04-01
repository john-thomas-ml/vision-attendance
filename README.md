# Automated Attendance System

This project implements an automated attendance tracking system using real-time face detection and recognition. It leverages a fine-tuned FaceNet model for face embeddings, YOLOv8 for face detection, and a Flask web application to display a live video feed and attendance statistics.

## Overview

The system consists of two main components:

1. **Model Training (train_facenet.py)**  
   - **FaceNet Fine-Tuning:** A script to fine-tune a pre-trained FaceNet model on a custom dataset of student images. Data augmentation (random flips, rotations, color jitter) is applied to improve model robustness.
   - **Reference Embeddings Generation:** The fine-tuned model is used to generate and save reference face embeddings for each student. These embeddings are later used for recognition.

2. **Real-Time Attendance Tracking (app.py & index.html)**  
   - **Face Detection & Recognition:** Uses YOLOv8 (with a face-specific model) to detect faces in a video stream and the fine-tuned FaceNet to extract embeddings. Cosine distance is computed to match live faces with stored reference embeddings.
   - **Attendance Database:** Tracks attendance using an SQLite database. Each student’s accumulated presence time is updated and once a threshold is reached, the student is marked as present.
   - **Web Interface:** A Flask-based web server provides a live video feed and a dashboard showing student timers and attendance status. The front-end is implemented using HTML, CSS, and JavaScript.

## Features

- **Face Detection:** YOLOv8 is used for fast and accurate face detection.
- **Face Recognition:** A fine-tuned FaceNet model extracts embeddings to recognize students.
- **Real-Time Processing:** Processes live video feed from a webcam and updates attendance continuously.
- **Attendance Tracking:** Uses an SQLite database to maintain records and mark attendance once a configurable time threshold is reached.
- **Web Dashboard:** Provides an intuitive web interface to start/stop the camera, view the live feed, and monitor attendance data.
- **Customizable Settings:** Threshold values and other parameters (e.g., consistency frames, cosine similarity threshold) can be adjusted.

## Prerequisites

- Python 3.7 or higher
- PyTorch
- facenet-pytorch
- OpenCV
- torchvision
- ultralytics (for YOLOv8)
- Flask
- SQLite3 (Python’s built-in module)
- PIL (Pillow)
- NumPy

You can install the required Python packages via pip:

```bash
pip install torch torchvision facenet-pytorch opencv-python ultralytics flask pillow numpy
```

## Project Structure

```
├── dataset/                 # Folder containing sub-folders of student images (e.g., John, Nelda, etc.)
├── fine_tuned_facenet.pth   # Fine-tuned FaceNet model (generated after training)
├── reference_embeddings.pkl # Saved reference embeddings for each student (generated during attendance setup)
├── train_facenet.py         # Script for fine-tuning the FaceNet model on your dataset and generating reference embeddings
├── app.py                   # Flask web application for real-time attendance tracking
├── templates/
│   └── index.html           # Front-end HTML template for the web dashboard
└── README.md                # This file
```

## Training the Model

1. **Fine-Tune FaceNet:**

   Run the training script to fine-tune the FaceNet model on your dataset. The script applies data augmentation and limits the number of images per class to balance the dataset.

   ```bash
   python train_facenet.py
   ```

   This will generate the fine-tuned model file (`fine_tuned_facenet.pth`).

2. **Generate Reference Embeddings:**

   The web application (app.py) uses the fine-tuned model to generate reference embeddings for each student. Ensure that your dataset is properly organized before running the web application.

## Running the Attendance System

Start the Flask web application to use the real-time web interface:

```bash
python app.py
```

- Open your web browser and navigate to [http://localhost:3001](http://localhost:3001) to access the dashboard.
- Use the on-screen buttons to start/stop the camera feed and monitor attendance data in real time.
- Adjust the attendance threshold and clear records using the provided controls.

## Customization

- **Thresholds & Parameters:**  
  You can modify parameters such as the attendance threshold, cosine similarity threshold, and consistency frames directly in the source files (`train_facenet.py` and `app.py`).
- **Data Augmentation:**  
  Adjust the transformations in `train_facenet.py` to suit your dataset and improve training results.
- **Web Interface:**  
  The front-end can be customized by editing `templates/index.html`.

## Troubleshooting

- **Webcam Issues:**  
  Ensure your webcam is connected and not used by another application.
- **Model Loading Errors:**  
  Verify that `fine_tuned_facenet.pth` exists and is in the correct path specified in your scripts.
- **Dataset Problems:**  
  Confirm that your dataset folder structure and image formats match the expected format.

## Acknowledgements

- [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
