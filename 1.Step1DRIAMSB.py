import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertModel, BertConfig
from torch.amp import GradScaler, autocast

# Configuration
EPOCHS = 10
BATCH_SIZE = 2  # Further reduced for local GPU
LEARNING_RATE = 1e-4
MASK_PROB = 0.15
MAX_SEQ_LENGTH = 512
SLIDING_STEP = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Masking function
def mask_peaks(intensities):
    mask = torch.rand(intensities.shape) < MASK_PROB
    masked_intensities = intensities.clone()
    masked_intensities[mask] = 0
    return masked_intensities, mask

# GPU Memory Check
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Load the ID file and spectrum files
class MassSpecDataset(Dataset):
    def __init__(self, id_file, spectra_dir):
        self.id_data = pd.read_csv(id_file)
        self.spectra_dir = spectra_dir
        self.file_ids = self.id_data['code'].tolist()

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        file_path = os.path.join(self.spectra_dir, f"{file_id}.txt")
        if not os.path.exists(file_path):
            return []

        with open(file_path, 'r') as f:
            lines = f.readlines()
            data = []
            for line in lines:
                if line.startswith('#'):
                    continue
                try:
                    mz, intensity = map(float, line.strip().split())
                    data.append((mz, intensity))
                except ValueError:
                    continue

        if len(data) == 0:
            return []

        mz_values, intensity_values = zip(*data)
        mz_tensor = torch.tensor(mz_values, dtype=torch.float16)
        intensity_tensor = torch.tensor(intensity_values, dtype=torch.float16)

        chunks = []
        for start in range(0, len(mz_tensor), SLIDING_STEP):
            end = min(start + MAX_SEQ_LENGTH, len(mz_tensor))
            chunk_mz = mz_tensor[start:end]
            chunk_intensity = intensity_tensor[start:end]
            chunks.append((chunk_mz, chunk_intensity))
        return chunks

# Custom collate function to handle variable-length chunks
def collate_fn(batch):
    all_chunks = []
    for chunks in batch:
        if not chunks:
            continue
        for chunk_mz, chunk_intensity in chunks:
            length = len(chunk_mz)
            attention_mask = torch.ones(length, dtype=torch.float16)
            all_chunks.append((chunk_mz.long(), chunk_intensity, attention_mask))
    if not all_chunks:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    mz_batch, intensity_batch, attention_batch = zip(*all_chunks)
    mz_tensor = torch.nn.utils.rnn.pad_sequence(mz_batch, batch_first=True)
    intensity_tensor = torch.nn.utils.rnn.pad_sequence(intensity_batch, batch_first=True)
    attention_tensor = torch.nn.utils.rnn.pad_sequence(attention_batch, batch_first=True)
    return mz_tensor, intensity_tensor, attention_tensor

# Model definition
class MassSpecFormer(nn.Module):
    def __init__(self):
        super(MassSpecFormer, self).__init__()
        config = BertConfig(hidden_size=64, num_attention_heads=2, num_hidden_layers=2)
        self.transformer = BertModel(config)
        self.regressor = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        prediction = self.regressor(outputs.last_hidden_state).squeeze(-1)
        return prediction

# Training function
def train(model, dataloader, optimizer, criterion):
    model.train()
    scaler = GradScaler()
    for epoch in range(EPOCHS):
        for mz, intensity, attention_mask in dataloader:
            if mz.numel() == 0:
                continue
            mz, intensity, attention_mask = mz.to(DEVICE), intensity.to(DEVICE), attention_mask.to(DEVICE)
            masked_intensity, mask = mask_peaks(intensity)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(mz, attention_mask=attention_mask)
                loss = criterion(outputs[mask], intensity[mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
            print_gpu_memory()

# Main
if __name__ == "__main__":
    id_file = "./DRIAMS-B/id/2018/2018_clean.csv"
    spectra_dir = "./DRIAMS-B/preprocessed/2018/"
    dataset = MassSpecDataset(id_file, spectra_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = MassSpecFormer().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train(model, dataloader, optimizer, criterion)
    torch.save(model.state_dict(), "massspecformer_pretrained.pth")
    print("Pre-training completed and model saved.")

