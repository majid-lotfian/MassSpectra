import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1️⃣ Dataset Preparation (For New Dataset)
# -------------------------------
class MassSpectraDataset(Dataset):
    def __init__(self, spectra_dir, metadata_file, max_length=6000, num_classes=56):
        self.spectra_dir = spectra_dir
        self.metadata = pd.read_csv(metadata_file)  # Load metadata
        self.max_length = max_length
        self.num_classes = num_classes  # Fixed model output size (56)

        # 🔹 Detect antibiotic resistance columns dynamically
        self.antibiotic_columns = [col for col in self.metadata.columns if col not in ["species", "code", "combined_code"]]
        print(f"🔎 Found {len(self.antibiotic_columns)} antibiotic resistance columns.")

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

        # 🔹 Pad labels to match model's fixed output size (56)
        padded_labels = torch.zeros(self.num_classes, dtype=torch.float32)
        padded_labels[:len(resistance_labels)] = resistance_tensor  # Copy existing labels

        return spectrum_tensor, padded_labels

# -------------------------------
# 2️⃣ Define MassSpecFormer Model
# -------------------------------
class MassSpecFormer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=6, num_classes=56):  # Start with 56 labels
        super(MassSpecFormer, self).__init__()

        # 🔹 Linear layer to project (m/z, intensity) into a 128D embedding space
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)  # Fixed at 56 labels

    def forward(self, x):
        """Pass (m/z, intensity) values through embedding and transformer."""
        x = self.embedding(x)  # 🔹 Convert (batch_size, seq_length, 2) → (batch_size, seq_length, 128)
        outputs = self.transformer(x)
        cls_token = outputs[:, 0, :]  # Take first token as CLS token
        return torch.sigmoid(self.fc(cls_token))  # Sigmoid for multi-label classification

# -------------------------------
# 3️⃣ Load Pretrained Model (With 56 Labels)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# Load pre-trained model (already trained on 56 labels)
model = MassSpecFormer(num_classes=56).to(device)

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
# 4️⃣ Load New Dataset (With Masking)
# -------------------------------
new_spectra_dir = "./DRIAMS-D/preprocessed/2018/"
new_metadata_file = "./DRIAMS-D/id/2018/2018_clean.csv"
new_dataset = MassSpectraDataset(new_spectra_dir, new_metadata_file, num_classes=56)

new_dataloader = DataLoader(new_dataset, batch_size=16, shuffle=True, drop_last=True)

# -------------------------------
# 5️⃣ Fine-Tune Model on New Dataset (With Masking)
# -------------------------------
criterion = nn.BCEWithLogitsLoss(reduction='none')  # Use element-wise loss
optimizer = optim.Adam(model.parameters(), lr=5e-5)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for spectra, labels in new_dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectra)

        # 🔹 Create mask to ignore missing labels (if all zeros, do not calculate loss)
        mask = (labels.sum(dim=0) > 0).float().to(device)  # 1 if label exists, 0 otherwise
        loss = criterion(outputs, labels) * mask  # Apply mask
        loss = loss.sum() / mask.sum()  # Normalize by the number of valid labels

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "./trained_models/massspecformer_BCD.pth")
print("✅ Model fine-tuned and saved as 'massspecformer_BCD.pth'")
