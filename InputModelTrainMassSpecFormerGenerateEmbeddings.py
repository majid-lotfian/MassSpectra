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
    def __init__(self, spectra_dir, metadata_file, max_length=6000):
        self.spectra_dir = spectra_dir
        self.metadata = pd.read_csv(metadata_file).iloc[:2386]  # Adjust based on new dataset
        self.max_length = max_length

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

        return spectrum_tensor, resistance_tensor

# -------------------------------
# 2️⃣ Define MassSpecFormer Model
# -------------------------------
class MassSpecFormer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=6, num_classes=44):  # 44 antibiotics
        super(MassSpecFormer, self).__init__()

        # 🔹 Linear layer to project (m/z, intensity) into a 128D embedding space
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Input: (m/z, intensity) → Output: 128D

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, num_classes)  # Multi-label classification

    def forward(self, x):
        """Pass (m/z, intensity) values through embedding and transformer."""
        x = self.embedding(x)  # 🔹 Convert (batch_size, seq_length, 2) → (batch_size, seq_length, 128)
        
        outputs = self.transformer(x)  # Pass through transformer
        cls_token = outputs[:, 0, :]  # Take first token as CLS token
        return torch.sigmoid(self.fc(cls_token))  # Sigmoid for multi-label classification

# -------------------------------
# 3️⃣ Load Pretrained Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# Load model architecture
num_antibiotics = 44
model = MassSpecFormer(num_classes=num_antibiotics).to(device)

# Load pretrained weights
model.load_state_dict(torch.load("./trained_models/massspecformer.pth"))
print("✅ Loaded pretrained MassSpecFormer model!")

# -------------------------------
# 4️⃣ Load New Dataset & DataLoader
# -------------------------------
new_spectra_dir = "./DRIAMS-C/preprocessed/2018/"
new_metadata_file = "./DRIAMS-C/id/2018/2018_clean.csv"

# Create dataset and dataloader for new data
new_dataset = MassSpectraDataset(new_spectra_dir, new_metadata_file)
new_dataloader = DataLoader(new_dataset, batch_size=16, shuffle=True, drop_last=True)

# -------------------------------
# 5️⃣ Continue Training on New Dataset
# -------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # 🔹 Lower LR for fine-tuning

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for spectra, labels in new_dataloader:
        spectra, labels = spectra.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectra)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "./trained_models/massspecformer_BC.pth")
print("✅ Model fine-tuned and saved as 'massspecformer_BC.pth'")

# -------------------------------
# 6️⃣ Generate Fine-Tuned Embeddings
# -------------------------------
class EmbeddingExtractor(nn.Module):
    """Extracts embeddings from MassSpecFormer without classification layer."""
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.embedding = model.embedding
        self.transformer = model.transformer

    def forward(self, x):
        x = self.embedding(x)
        outputs = self.transformer(x)
        cls_embedding = outputs[:, 0, :]
        return cls_embedding

# Load fine-tuned model
embedding_model = EmbeddingExtractor(model).to(device)
embedding_model.eval()

# Generate embeddings
embeddings_dict = {}
with torch.no_grad():
    for i, (spectra, _) in enumerate(new_dataloader):
        spectra = spectra.to(device)
        embeddings = embedding_model(spectra)
        for j, emb in enumerate(embeddings):
            embeddings_dict[f"spectrum_{i * len(embeddings) + j}"] = emb.cpu().numpy()

# Save embeddings
torch.save(embeddings_dict, "./embeddings/massspecformer_BC_embeddings.pt")
print("✅ Fine-tuned embeddings saved as 'massspecformer_BC_embeddings.pt'")
