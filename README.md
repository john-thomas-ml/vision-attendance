# Automated Attendance System

This project implements an automated attendance tracking system using real-time face detection and recognition. It leverages a fine-tuned FaceNet model for face embeddings, YOLOv8 for face detection, and a Flask web application to display a live video feed and attendance statistics.

## Overview

The system consists of two main components:

1. **Model Training (train_facenet.py)**  
   - **FaceNet Fine-Tuning:** A script to fine-tune a pre-trained FaceNet model on a custom dataset of student images. Aggressive data augmentation (random flips, rotations, color jitter, perspective distortion, etc.) is applied to improve model robustness.  
   - **Enhanced Training Features:** Includes early stopping, computation of confusion matrices, precision/recall/F1 metrics, and logging of training/validation metrics to a CSV file (`training_metrics.csv`). Visualizations such as loss/accuracy plots and confusion matrices are generated and saved in the `visualizations/` directory.  
   - **Evaluation Phase:** After training, the script evaluates the model on a test set using embeddings (similar to the recognition process in `app.py`) and generates a test confusion matrix and metrics.  
   - **Reference Embeddings Generation:** The fine-tuned model is used to generate and save reference face embeddings for each student. These embeddings are later used for recognition.

2. **Real-Time Attendance Tracking (app.py & index.html)**  
   - **Face Detection & Recognition:** Uses YOLOv8 (with a face-specific model) to detect faces in a video stream and the fine-tuned FaceNet to extract embeddings. Cosine distance is computed to match live faces with stored reference embeddings.  
   - **Attendance Database:** Tracks attendance using an SQLite database. Each student’s accumulated presence time is updated, and once a configurable time threshold is reached, the student is marked as present.  
   - **Web Interface:** A Flask-based web server provides a live video feed and a dashboard showing student timers and attendance status. The front-end is implemented using HTML, CSS, and JavaScript.

## Features

- **Face Detection:** YOLOv8 is used for fast and accurate face detection.  
- **Face Recognition:** A fine-tuned FaceNet model extracts embeddings to recognize students.  
- **Real-Time Processing:** Processes live video feed from a webcam and updates attendance continuously.  
- **Attendance Tracking:** Uses an SQLite database to maintain records and mark attendance once a configurable time threshold is reached.  
- **Web Dashboard:** Provides an intuitive web interface to start/stop the camera, view the live feed, and monitor attendance data.  
- **Customizable Settings:** Threshold values and other parameters (e.g., consistency frames, cosine similarity threshold) can be adjusted.  
- **Training Visualizations:** Generates visualizations such as confusion matrices, loss, and accuracy plots to evaluate model performance during training.  
- **Metrics Logging:** Logs training, validation, and test metrics (e.g., accuracy, precision, recall, F1) for detailed analysis.

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
- Matplotlib (for visualizations)  
- Seaborn (for enhanced visualization plots)  

You can install the required Python packages via pip:

```bash
pip install torch torchvision facenet-pytorch opencv-python ultralytics flask pillow numpy matplotlib seaborn
```

## Project Structure

```
├── dataset/                 # Folder containing sub-folders of student images (e.g., John, Nelda, etc.)
│   ├── John/               # Images for student John
│   ├── Nelda/              # Images for student Nelda
│   ├── Parvathy/           # Images for student Parvathy
│   ├── Safran/             # Images for student Safran
├── templates/              # Folder containing HTML templates
│   └── index.html          # Front-end HTML template for the web dashboard
├── visualizations/         # Folder containing generated visualizations (e.g., confusion matrices, loss/accuracy plots)
├── app.py                  # Flask web application for real-time attendance tracking
├── attendance.db           # SQLite database for storing attendance records
├── fine_tuned_facenet.pth  # Fine-tuned FaceNet model (generated after training)
├── reference_embeddings.pkl # Saved reference embeddings for each student (generated during attendance setup)
├── train_facenet.py        # Script for fine-tuning the FaceNet model, evaluating, and generating reference embeddings
├── training_metrics.csv    # CSV file logging training, validation, and test metrics
├── yolov8n-face.pt         # YOLOv8 face detection model
└── README.md               # This file
```

## Training the Model

1. **Prepare the Dataset:**  
   Ensure your dataset is organized in the `dataset/` directory with subfolders named after each student (e.g., `dataset/John/`, `dataset/Nelda/`). Each subfolder should contain images in `.jpg` or `.png` format.

2. **Fine-Tune FaceNet and Evaluate:**  
   Run the training script to fine-tune the FaceNet model on your dataset. The script applies data augmentation, limits the number of images per class to balance the dataset, and evaluates the model using embeddings on a test set. It also generates visualizations and logs metrics.

   ```bash
   python train_facenet.py
   ```

   This will generate the following:  
   - `fine_tuned_facenet.pth`: The fine-tuned FaceNet model.  
   - `visualizations/`: Directory containing plots such as confusion matrices, loss, and accuracy over epochs.  
   - `training_metrics.csv`: A CSV file logging training, validation, and test metrics.

3. **Generate Reference Embeddings:**  
   The web application (`app.py`) uses the fine-tuned model to generate reference embeddings for each student (`reference_embeddings.pkl`). These are created automatically when you run the web application if they do not already exist.

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
- **Visualizations:**  
  Modify the visualization code in `train_facenet.py` to generate additional plots or change their appearance.

## Troubleshooting

- **Webcam Issues:**  
  Ensure your webcam is connected and not used by another application.  
- **Model Loading Errors:**  
  Verify that `fine_tuned_facenet.pth` and `yolov8n-face.pt` exist and are in the correct paths specified in your scripts.  
- **Dataset Problems:**  
  Confirm that your dataset folder structure and image formats match the expected format (subfolders for each student with `.jpg` or `.png` images).  
- **Visualization Errors:**  
  Ensure `matplotlib` and `seaborn` are installed, and the `visualizations/` directory has write permissions.  
- **Metrics Logging Issues:**  
  Verify that the script has write permissions to create `training_metrics.csv`.

## Acknowledgements

- [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch)  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)