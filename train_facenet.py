import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Parameters
data_dir = "C:/Users/tough/OneDrive/Documents/Tools/Coding/Python/Projects/Automated Attendance/dataset"
batch_size = 16
image_size = 160
num_epochs = 20
num_classes = 4
max_images_per_class = 20  # 20 images per person

# Create a directory to save visualizations
vis_dir = "C:/Users/tough/OneDrive/Documents/Tools/Coding/Python/Projects/Automated Attendance/visualizations"
os.makedirs(vis_dir, exist_ok=True)

# Custom Dataset for FaceNet
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images_per_class = max_images_per_class
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.images = []
        print(f"Root directory: {root_dir}")
        print(f"Found classes: {self.classes}")
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            print(f"Checking class directory: {class_dir}")
            valid_extensions = ('.jpg', '.png')
            class_images = [
                os.path.join(class_dir, img_name)
                for img_name in os.listdir(class_dir)
                if img_name.lower().endswith(valid_extensions)
            ]
            print(f"Found images in {cls}: {len(class_images)}")
            if self.max_images_per_class is not None:
                if len(class_images) > self.max_images_per_class:
                    print(f"Limiting images for {cls} to {self.max_images_per_class}")
                    class_images = class_images[:self.max_images_per_class]
                elif len(class_images) < self.max_images_per_class:
                    print(f"Warning: Class {cls} has only {len(class_images)} images, less than max_images_per_class ({self.max_images_per_class}).")

            for img_path in class_images:
                self.images.append((img_path, self.class_to_idx[cls]))
        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes: {self.classes}")
        img_counts = {cls: 0 for cls in self.classes}
        for _, label_idx in self.images:
            label_cls = self.classes[label_idx]
            img_counts[label_cls] += 1
        print(f"Final image counts per class used for training: {img_counts}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading or transforming image {img_path}: {e}")
            dummy_img = torch.zeros((3, image_size, image_size))
            return dummy_img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
    ], p=0.5),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3.0))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Transform for test set (no augmentation, just preprocessing)
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("--- Using Aggressive Data Augmentation Pipeline for Training ---")
print(transform)
print("----------------------------------------------------")

# Load dataset
dataset = FaceDataset(root_dir=data_dir, transform=transform, max_images_per_class=max_images_per_class)

# Check if dataset is empty
if len(dataset) == 0:
    raise ValueError(f"No images found in the dataset directory '{data_dir}' with the specified criteria.")

# Split dataset into training, validation, and test sets (70% train, 15% val, 15% test)
indices = list(range(len(dataset)))
labels = [label for _, label in dataset.images]  # Extract labels for stratification

# First split: 70% train, 30% (val + test)
train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.3,  # 30% for val + test
    random_state=42,
    stratify=labels
)

# Extract labels for temp_indices
temp_labels = [labels[idx] for idx in temp_indices]

# Second split: Split the 30% into 15% val and 15% test
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.5,  # 50% of 30% = 15% of total
    random_state=42,
    stratify=temp_labels
)

print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test")

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)
# Test loader will use test_transform
test_dataset = FaceDataset(root_dir=data_dir, transform=test_transform, max_images_per_class=max_images_per_class)
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0)

# Load pre-trained Facenet model
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
print(f"Loaded InceptionResnetV1 (pretrained='vggface2') with {num_classes} output classes.")
model.train()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Function to compute accuracy
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Function to compute confusion matrix
def compute_confusion_matrix(outputs, labels, num_classes, idx_to_class):
    _, predicted = torch.max(outputs, 1)
    conf_matrix = defaultdict(lambda: defaultdict(int))
    for pred, true in zip(predicted, labels):
        pred_class = idx_to_class[pred.item()]
        true_class = idx_to_class[true.item()]
        conf_matrix[true_class][pred_class] += 1
    return conf_matrix

# Function to compute precision, recall, and F1 score from confusion matrix
def compute_metrics(conf_matrix, classes):
    metrics = {}
    for true_class in classes:
        tp = conf_matrix[true_class][true_class] if true_class in conf_matrix else 0
        fp = sum(conf_matrix[other][true_class] for other in classes if other != true_class and other in conf_matrix and true_class in conf_matrix[other])
        fn = sum(conf_matrix[true_class][other] for other in classes if other != true_class and true_class in conf_matrix and other in conf_matrix[true_class])
        total_true = sum(conf_matrix[true_class].values()) if true_class in conf_matrix else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[true_class] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': total_true}
    return metrics

# Function to plot and save confusion matrix
def plot_confusion_matrix(conf_matrix, classes, title, save_path):
    # Convert confusion matrix to a numpy array
    matrix = np.zeros((len(classes), len(classes)))
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = conf_matrix[true_class][pred_class] if pred_class in conf_matrix[true_class] else 0

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Early stopping parameters
patience = 2  # Number of epochs to wait for improvement in validation loss
best_val_loss = float("inf")
epochs_no_improve = 0
best_model_state = None
early_stop_epoch = num_epochs  # Will be updated if early stopping occurs

# Initialize CSV file for logging metrics
metrics_file = "training_metrics.csv"
with open(metrics_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])

print(f"Starting training for {num_epochs} epochs with early stopping (patience={patience})...")
# Training loop with accuracy, confusion matrix, and early stopping
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    processed_batches = 0
    train_conf_matrix = defaultdict(lambda: defaultdict(int))

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if inputs is None or inputs.nelement() == 0:
            print(f"Warning: Skipping empty or invalid batch at epoch {epoch+1}, step {i+1}")
            continue

        optimizer.zero_grad()
        outputs = model(inputs)

        if outputs is None or labels is None:
            print(f"Warning: Skipping batch due to None outputs or labels at epoch {epoch+1}, step {i+1}")
            continue

        try:
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Compute accuracy
            batch_accuracy = compute_accuracy(outputs, labels)
            running_corrects += batch_accuracy * labels.size(0)
            total_samples += labels.size(0)

            # Update confusion matrix
            batch_conf_matrix = compute_confusion_matrix(outputs, labels, num_classes, dataset.idx_to_class)
            for true_class in batch_conf_matrix:
                for pred_class in batch_conf_matrix[true_class]:
                    train_conf_matrix[true_class][pred_class] += batch_conf_matrix[true_class][pred_class]

            processed_batches += 1
        except Exception as e:
            print(f"Error during forward/backward pass at epoch {epoch+1}, step {i+1}: {e}")
            continue

        print_freq = max(1, len(train_loader) // 5)
        if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
            if processed_batches > 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Avg Loss: {running_loss/processed_batches:.4f}, "
                      f"Train Accuracy: {running_corrects/total_samples:.4f}")

    # Calculate epoch training metrics
    if processed_batches > 0:
        epoch_loss = running_loss / processed_batches
        train_accuracy = running_corrects / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        print(f"--- End of Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f} ---")
        print("Training Confusion Matrix:")
        for true_class in dataset.classes:
            print(f"True {true_class}: {dict(train_conf_matrix[true_class])}")
        train_metrics = compute_metrics(train_conf_matrix, dataset.classes)
        print("Training Metrics:")
        for cls, metric in train_metrics.items():
            print(f"{cls}: Precision={metric['precision']:.4f}, Recall={metric['recall']:.4f}, F1={metric['f1']:.4f}, Support={metric['support']}")
        
        # Plot training confusion matrix
        plot_confusion_matrix(
            train_conf_matrix, 
            dataset.classes, 
            f"Training Confusion Matrix - Epoch {epoch+1}", 
            os.path.join(vis_dir, f"train_conf_matrix_epoch_{epoch+1}.png")
        )
    else:
        print(f"--- End of Epoch {epoch+1}, No batches successfully processed ---")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_samples = 0
    val_batches = 0
    val_conf_matrix = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            if inputs is None or inputs.nelement() == 0:
                continue

            outputs = model(inputs)
            if outputs is None or labels is None:
                continue

            try:
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                batch_accuracy = compute_accuracy(outputs, labels)
                val_corrects += batch_accuracy * labels.size(0)
                val_samples += labels.size(0)

                # Update confusion matrix
                batch_conf_matrix = compute_confusion_matrix(outputs, labels, num_classes, dataset.idx_to_class)
                for true_class in batch_conf_matrix:
                    for pred_class in batch_conf_matrix[true_class]:
                        val_conf_matrix[true_class][pred_class] += batch_conf_matrix[true_class][pred_class]

                val_batches += 1
            except Exception as e:
                print(f"Error during validation at epoch {epoch+1}, step {i+1}: {e}")
                continue

    if val_batches > 0:
        val_loss = val_loss / val_batches
        val_accuracy = val_corrects / val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"--- Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f} ---")
        print("Validation Confusion Matrix:")
        for true_class in dataset.classes:
            print(f"True {true_class}: {dict(val_conf_matrix[true_class])}")
        val_metrics = compute_metrics(val_conf_matrix, dataset.classes)
        print("Validation Metrics:")
        for cls, metric in val_metrics.items():
            print(f"{cls}: Precision={metric['precision']:.4f}, Recall={metric['recall']:.4f}, F1={metric['f1']:.4f}, Support={metric['support']}")
        
        # Plot validation confusion matrix
        plot_confusion_matrix(
            val_conf_matrix, 
            dataset.classes, 
            f"Validation Confusion Matrix - Epoch {epoch+1}", 
            os.path.join(vis_dir, f"val_conf_matrix_epoch_{epoch+1}.png")
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model state
            epochs_no_improve = 0
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                early_stop_epoch = epoch + 1
                break
    else:
        print(f"--- No valid validation batches processed ---")

    # Log metrics to CSV
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, epoch_loss, train_accuracy, val_loss, val_accuracy])

# Restore the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored best model from epoch {best_epoch} with validation loss {best_val_loss:.4f}")

# Plot training and validation loss over epochs (up to early_stop_epoch)
plt.figure(figsize=(10, 5))
plt.plot(range(1, early_stop_epoch + 1), train_losses, label='Training Loss')
plt.plot(range(1, early_stop_epoch + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid()
plt.savefig(os.path.join(vis_dir, "loss_over_epochs.png"))
plt.close()

# Plot training and validation accuracy over epochs (up to early_stop_epoch)
plt.figure(figsize=(10, 5))
plt.plot(range(1, early_stop_epoch + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, early_stop_epoch + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid()
plt.savefig(os.path.join(vis_dir, "accuracy_over_epochs.png"))
plt.close()

# Save the best model
model.logits = torch.nn.Identity()
model.eval()
save_path = "C:/Users/tough/OneDrive/Documents/Tools/Coding/Python/Projects/Automated Attendance/fine_tuned_facenet.pth"
try:
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model evaluation features saved to {save_path}")
except Exception as e:
    print(f"\nError saving model state_dict: {e}")

# Test phase: Evaluate using embeddings (similar to app.py)
print("\n--- Evaluating on Test Set Using Embeddings ---")

# Function to compute cosine distance
def cosine_distance(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return 1 - cosine_similarity

# Generate reference embeddings from training set
reference_embeddings = {}
COSINE_THRESHOLD = 0.285  # Same as in app.py
for cls in dataset.classes:
    cls_indices = [idx for idx, (_, label) in enumerate(dataset.images) if label == dataset.class_to_idx[cls] and idx in train_indices]
    embeddings = []
    for idx in cls_indices:
        img, _ = dataset[idx]
        img = img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = model(img).numpy().flatten()
        embeddings.append(embedding)
    reference_embeddings[cls] = embeddings
print("Generated reference embeddings from training set.")

# Evaluate on test set
test_correct = 0
test_total = 0
test_conf_matrix = defaultdict(lambda: defaultdict(int))

with torch.no_grad():
    for inputs, labels in test_loader:
        if inputs is None or inputs.nelement() == 0:
            continue

        # Generate embeddings for test images
        embeddings = model(inputs).numpy()

        for idx, embedding in enumerate(embeddings):
            embedding = embedding.flatten()
            true_label = dataset.idx_to_class[labels[idx].item()]
            min_dist = float("inf")
            predicted_class = None

            # Compare with reference embeddings
            for cls, ref_embs in reference_embeddings.items():
                for ref_emb in ref_embs:
                    dist = cosine_distance(embedding, ref_emb)
                    if dist < min_dist and dist < COSINE_THRESHOLD:
                        min_dist = dist
                        predicted_class = cls

            predicted_label = predicted_class if predicted_class else "Unknown"
            test_conf_matrix[true_label][predicted_label] += 1

            if predicted_label == true_label:
                test_correct += 1
            test_total += 1

# Compute test accuracy and metrics
test_accuracy = test_correct / test_total if test_total > 0 else 0
print(f"Test Accuracy (Embedding-based): {test_accuracy:.4f} ({test_correct}/{test_total})")
print("Test Confusion Matrix:")
for true_class in dataset.classes:
    print(f"True {true_class}: {dict(test_conf_matrix[true_class])}")
test_metrics = compute_metrics(test_conf_matrix, dataset.classes)
print("Test Metrics:")
for cls, metric in test_metrics.items():
    print(f"{cls}: Precision={metric['precision']:.4f}, Recall={metric['recall']:.4f}, F1={metric['f1']:.4f}, Support={metric['support']}")

# Plot test confusion matrix
plot_confusion_matrix(
    test_conf_matrix, 
    dataset.classes + ["Unknown"],  # Include "Unknown" class for test phase
    "Test Confusion Matrix (Embedding-based)", 
    os.path.join(vis_dir, "test_conf_matrix.png")
)