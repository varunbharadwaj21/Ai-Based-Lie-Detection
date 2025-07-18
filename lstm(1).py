# -*- coding: utf-8 -*-

# **LSTM(1)** trail 1 
"""

# Imports and Setup
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
from google.colab import drive


# Mounting Google Drive

drive.mount('')

# Paths and Parameters

base_path = "path"
model_save_path = os.path.join(base_path, "saved_models")
os.makedirs(model_save_path, exist_ok=True)

frames_path = os.path.join(base_path, "frames")
specs_path = os.path.join(base_path, "spectrograms")
split_df = pd.read_csv(os.path.join(base_path, "split_clips.csv"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
NUM_FRAMES = 5

# Dataset for Fusion

class FusionDataset(Dataset):
    def __init__(self, df, split, transform=None):
        self.samples = df[df['split'] == split].reset_index(drop=True)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        file = row['file_name']
        label = 0 if row['label'] == 'truth' else 1

        frame_dir = os.path.join(frames_path, self.split, row['label'], file)
        spec_path = os.path.join(specs_path, self.split, row['label'], f"{file}.png")

        frames = []
        for i in range(NUM_FRAMES):
            frame_path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert('RGB')
            else:
                frame = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)

        spec = Image.open(spec_path).convert('RGB')
        if self.transform:
            spec = self.transform(spec)

        return frames, spec, label

# Transforms and DataLoaders
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = FusionDataset(split_df, split='train', transform=train_transform)
val_ds = FusionDataset(split_df, split='val', transform=val_transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Building Models

vision_model = models.resnet18(pretrained=True)
vision_model.fc = nn.Identity()
vision_model = vision_model.to(device)

audio_model = models.resnet18(pretrained=True)
audio_model.fc = nn.Identity()
audio_model = audio_model.to(device)

class FusionLSTMNet(nn.Module):
    def __init__(self, vision_model, audio_model, hidden_size=256, num_layers=1):
        super(FusionLSTMNet, self).__init__()
        self.vision_model = vision_model
        self.audio_model = audio_model
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + 512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, frames, spec):
        batch_size, seq_len, c, h, w = frames.shape
        frames = frames.view(batch_size * seq_len, c, h, w)
        frame_features = self.vision_model(frames)
        frame_features = frame_features.view(batch_size, seq_len, -1)

        _, (hn, _) = self.lstm(frame_features)
        lstm_output = hn[-1]

        audio_features = self.audio_model(spec)

        combined = torch.cat((lstm_output, audio_features), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = FusionLSTMNet(vision_model, audio_model).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training and Validation Functions
def train_epoch():
    model.train()
    running_loss, correct = 0.0, 0
    for frames, specs, labels in tqdm(train_dl):
        frames, specs, labels = frames.to(device), specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(frames, specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * frames.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(train_dl.dataset), correct / len(train_dl.dataset)

def validate():
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for frames, specs, labels in val_dl:
            frames, specs, labels = frames.to(device), specs.to(device), labels.to(device)
            outputs = model(frames, specs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * frames.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return val_loss / len(val_dl.dataset), correct / len(val_dl.dataset)


# Training Loop

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()
    print(f"\nEpoch {epoch+1}/{EPOCHS}:")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

    save_path = os.path.join(model_save_path, f"fusion_lstm_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved at {save_path}")

print("\nTraining complete. Fusion LSTM model ready.")

# Save the fusion model
save_path = "model.pth"
torch.save(model.state_dict(), save_path)
print(f" Model saved at {save_path}")

# Imports and Setup
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import numpy as np
from google.colab import drive


# Mount Google Drive
#drive.mount('')


# Set Paths
base_path = "path"
frames_path = os.path.join(base_path, "frames")
specs_path = os.path.join(base_path, "spectrograms")
split_df = pd.read_csv(os.path.join(base_path, "split_clips.csv"))
model_path = os.path.join(base_path, "saved_models/fusion_lstm_epoch_10.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_FRAMES = 5

# Define Dataset Class

class FusionDataset(Dataset):
    def __init__(self, df, split, transform=None):
        self.samples = df[df['split'] == split].reset_index(drop=True)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        file = row['file_name']
        label = 0 if row['label'] == 'truth' else 1

        frame_dir = os.path.join(frames_path, self.split, row['label'], file)
        spec_path = os.path.join(specs_path, self.split, row['label'], f"{file}.png")

        frames = []
        for i in range(NUM_FRAMES):
            frame_path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert('RGB')
            else:
                frame = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)

        spec = Image.open(spec_path).convert('RGB')
        if self.transform:
            spec = self.transform(spec)

        return frames, spec, label


# Load Validation Dataset

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_ds = FusionDataset(split_df, split='val', transform=val_transform)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Define Fusion Model

vision_model = models.resnet18(pretrained=False)
vision_model.fc = nn.Identity()
vision_model = vision_model.to(device)

audio_model = models.resnet18(pretrained=False)
audio_model.fc = nn.Identity()
audio_model = audio_model.to(device)

class FusionLSTMNet(nn.Module):
    def __init__(self, vision_model, audio_model, hidden_size=256, num_layers=1):
        super(FusionLSTMNet, self).__init__()
        self.vision_model = vision_model
        self.audio_model = audio_model
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + 512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, frames, spec):
        batch_size, seq_len, c, h, w = frames.shape
        frames = frames.view(batch_size * seq_len, c, h, w)
        frame_features = self.vision_model(frames)
        frame_features = frame_features.view(batch_size, seq_len, -1)

        _, (hn, _) = self.lstm(frame_features)
        lstm_output = hn[-1]

        audio_features = self.audio_model(spec)

        combined = torch.cat((lstm_output, audio_features), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load Model

model = FusionLSTMNet(vision_model, audio_model).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate on Validation Set

y_true = []
y_pred = []

with torch.no_grad():
    for frames, specs, labels in tqdm(val_dl):
        frames, specs, labels = frames.to(device), specs.to(device), labels.to(device)
        outputs = model(frames, specs)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Confusion Matrix

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Truth', 'Deception'], yticklabels=['Truth', 'Deception'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(base_path, "confusion_matrix.png"))
plt.show()


# Classification Report

print(classification_report(y_true, y_pred, target_names=['Truth', 'Deception']))
