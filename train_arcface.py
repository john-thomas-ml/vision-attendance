import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import cv2
import os
from PIL import Image
import numpy as np

# Parameters
data_dir = "C:/dataset"
batch_size = 16  # Reduced for CPU
image_size = 160
num_epochs = 5  # Increased for better training
num_classes = 4  # Number of students (John, Nelda)
max_images_per_class = 13  # Limit to 13 images per student to balance the dataset

# Custom Dataset for ArcFace
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images_per_class = max_images_per_class
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        print(f"Root directory: {root_dir}")
        print(f"Found classes: {self.classes}")
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            print(f"Checking class directory: {class_dir}")
            # Accept both .jpg and .png (case-insensitive)
            valid_extensions = ('.jpg', '.png')
            class_images = [
                os.path.join(class_dir, img_name) 
                for img_name in os.listdir(class_dir) 
                if img_name.lower().endswith(valid_extensions)
            ]
            print(f"Found images in {cls}: {class_images}")
            if self.max_images_per_class is not None and len(class_images) > self.max_images_per_class:
                class_images = class_images[:self.max_images_per_class]
            for img_path in class_images:
                self.images.append((img_path, self.class_to_idx[cls]))
        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate images by up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly adjust brightness, contrast, saturation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset with the max_images_per_class limit
dataset = FaceDataset(root_dir=data_dir, transform=transform, max_images_per_class=max_images_per_class)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained ArcFace model
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
model.train()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}")
            running_loss = 0.0

# Remove the logits layer before saving
model.logits = torch.nn.Identity()
model.eval()

# Save the fine-tuned model without the logits layer
torch.save(model.state_dict(), "C:/fine_tuned_arcface.pth")
print("Training complete. Model saved to fine_tuned_arcface.pth")