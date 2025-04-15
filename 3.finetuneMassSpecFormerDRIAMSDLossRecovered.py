import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1️⃣ Dataset Preparation (With Dynamic Label Count)
# -------------------------------
class MassSpectraDataset(Dataset):
    def __init__(self, spectra_dir, metadata_file, max_length=6000):
        self.spectra_dir = spectra_dir
        self.metadata = pd.read_csv(metadata_file)  # Load metadata
        self.max_length = max_length

        # 🔹 Detect antibiotic resistance columns dynamically
        self.antibiotic_columns = [col for col in self.metadata.columns if col not in ["species", "code", "combined_code"]]
        self.num_classes = len(self.antibiotic_columns)  # Dynamic label count

        print(f"🔎 Found {self.num_classes} antibiotic resistance labels.")

        if not self.antibiotic_columns:
            raise KeyError("🚨 No antibiotic resistance columns found in metadata!")

        # Map 'S', 'R', 'I' to binary labels (Resistant=1, Otherwise=0)
        self.label_mapping = {'S': 0, 'I': 0, 'R': 1}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        spectrum_filename = row['code'] + '.txt'
        spectrum_path = os.path.join(self.spectra_dir, spectrum_filename)

        # 🔹 Check if spectrum file exists
        if not os.path.exists(spectrum_path):
            print(f"🚨 Missing spectrum file: {spectrum_path}. Skipping.")
            return None  # Skip missing files

        # Load spectrum data (skip first 3 metadata lines)
        spectrum = np.loadtxt(spectrum_path, skiprows=3)

        # Normalize intensity values
        spectrum[:, 1] /= np.max(spectrum[:, 1])

        # 🔹 Ensure all spectra have exactly 6000 bins
        if spectrum.shape[0] > self.max_length:
            spectrum = spectrum[:self.max_length]  # Truncate if too long
        elif spectrum.shape[0] < self.max_length:
            padding = np.zeros((self.max_length - spectrum.shape[0], 2))
            spectrum = np.vstack((spectrum, padding))  # Pad if too short

        # Convert to PyTorch tensor (both m/z and intensity)
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32)  # Shape: (6000, 2)

        # Convert antibiotic resistance labels into a binary vector
        resistance_labels = [self.label_mapping.get(row[antibiotic], 0) for antibiotic in self.antibiotic_columns]
        resistance_tensor = torch.tensor(resistance_labels, dtype=torch.float32)

        return spectrum_tensor, resistance_tensor

# -------------------------------
# 2️⃣ Define MassSpecFormer Model (With Flexible Output Layer)
# -------------------------------
class MassSpecFormer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=6, num_classes=56):  
        super(MassSpecFormer, self).__init__()

        # 🔹 Linear layer to project (m/z, intensity) into a 128D embedding space
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Output layer (this will be adjusted dynamically)
        self.fc = nn.Linear(hidden_dim, num_classes)  

    def forward(self, x):
        x = self.embedding(x)  # 🔹 Convert (batch_size, seq_length, 2) → (batch_size, seq_length, 128)
        outputs = self.transformer(x)
        cls_token = outputs[:, 0, :]  # Take first token as CLS token
        return torch.sigmoid(self.fc(cls_token))  # Sigmoid for multi-label classification

# -------------------------------
# 3️⃣ Load Pretrained Model (Adapted for Dynamic Labels)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# Load new dataset to determine label count
new_spectra_dir = "./DRIAMS-D/preprocessed/2018/"
new_metadata_file = "./DRIAMS-D/id/2018/2018_clean.csv"
new_dataset = MassSpectraDataset(new_spectra_dir, new_metadata_file)

num_classes = new_dataset.num_classes  # Adjust model output size dynamically
print(f"🔧 Adjusting model to {num_classes} output labels.")

# Initialize model with the correct number of classes
model = MassSpecFormer(num_classes=num_classes).to(device)

# Load trained model weights
pretrained_dict = torch.load("./trained_models/massspecformer_BC.pth")
model_dict = model.state_dict()

# Remove the last layer (fc) from the pre-trained weights
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "fc" not in k}

# Update the model's state dictionary with pre-trained weights
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)
print("✅ Loaded pre-trained weights (excluding the final classification layer).")

# -------------------------------
# 4️⃣ Load New Dataset
# -------------------------------
new_dataloader = DataLoader(new_dataset, batch_size=16, shuffle=True, drop_last=True)

# -------------------------------
# 5️⃣ Fine-Tune Model on New Dataset (With Gradual Unfreezing)
# -------------------------------
criterion = nn.BCEWithLogitsLoss(reduction='none')  
optimizer = optim.Adam(model.parameters(), lr=5e-5)

num_epochs = 20
freeze_epochs = 3  # 🔹 Freeze transformer for first 3 epochs

# 🔹 Initially freeze transformer layers
for param in model.transformer.parameters():
    param.requires_grad = False

for epoch in range(num_epochs):
    if epoch == freeze_epochs:
        print("🔓 Unfreezing transformer layers for full fine-tuning.")
        for param in model.transformer.parameters():
            param.requires_grad = True

    model.train()
    total_loss = 0

    for batch in new_dataloader:
        spectra, labels = batch
        spectra = spectra.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectra)

        # 🔹 Improved Masking (Per Sample)
        mask = (labels >= 0).float().to(device)  
        loss = (criterion(outputs, labels) * mask).sum() / mask.sum()  # Normalize properly

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "./trained_models/massspecformer_BCD.pth")
print("✅ Model fine-tuned and saved as 'massspecformer_BCD.pth'")
