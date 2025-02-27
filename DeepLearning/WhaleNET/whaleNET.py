# Libraries
import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
from zipfile import ZipFile 
from torchaudio.transforms import Resample, MelSpectrogram
import os
import numpy as np
from kymatio.torch import Scattering1D
import torch.nn as nn
from torch.nn.functional import pad
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Check device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

# ============================================= Parameters =============================================

train_csv = "Dataset/CSVs/FullTrain.csv"
val_csv = "Dataset/CSVs/FullValidation.csv"
test_csv = "Dataset/CSVs/FullTest.csv"

batch_size = 4
target_sr = 24000
target_length_seconds = 3
target_length_samples = target_sr * target_length_seconds

learning_rate = 0.01
weight_decay = 0.001

# ============================================= Data Loading =============================================

# Load dataset CSVs
def load_csv(csv_path):
    return pd.read_csv(csv_path)

df_train = load_csv(train_csv)
df_test = load_csv(val_csv)
df_val = load_csv(test_csv)

dataframes = {"train": df_train, "test": df_test, "val": df_val}

# Load audio files and labels
def load_audio_data(df, max_files=None):
    data_list, labels = [], []
    
    for i, row in enumerate(df.iterrows()):
        if max_files and i >= max_files:  # Optional limit for debugging
            break
        
        file_path = row[1]['File'].replace('../../', '')  
        label = row[1]['Class']
        
        try:
            waveform, sr = torchaudio.load(file_path)
            resampler = Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)  # Resample audio
            
            # Reduce memory footprint by downsampling
            if waveform.shape[1] > target_length_samples:
                waveform = waveform[:, :target_length_samples]  # Trim to target length
            else:
                waveform = pad(waveform, (0, target_length_samples - waveform.shape[1]))  # Pad if too short

            data_list.append(waveform)
            labels.append(label)
            
            if (i + 1) % 100 == 0:  # Print progress every 100 files
                print(f"Loaded {i+1}/{len(df)} files...")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data_list, labels

# Load datasets in smaller chunks
train_data, train_labels = load_audio_data(df_train)  # Adjust max_files if needed
test_data, test_labels = load_audio_data(df_test)
val_data, val_labels = load_audio_data(df_val)

# check if all audio files are accessible
print(f"Train: {len(train_data)} samples")
print(f"Test: {len(test_data)} samples")
print(f"Validation: {len(val_data)} samples")

# Standardization
def standardize(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    std = torch.std(X, dim=1, keepdim=True)
    return (X - mean) / std

# Convert lists to tensors
def process_data(data_list):
    standardized_data = [standardize(x) for x in data_list]
    padded_data = [torch.nn.functional.pad(x, (0, max(0, target_length_samples - x.shape[1]))) for x in standardized_data]
    return torch.stack(padded_data)

X_train = process_data(train_data)
X_test = process_data(test_data)
X_val = process_data(val_data)

# Encode labels
label_encoder = LabelEncoder()
y_train = torch.tensor(label_encoder.fit_transform(train_labels))
y_test = torch.tensor(label_encoder.transform(test_labels))
y_val = torch.tensor(label_encoder.transform(val_labels))

# Feature extraction (Mel Spectrogram)
spectr = MelSpectrogram(normalized=True, n_mels=64).to(device)

def extract_features(X):
    N_batch = (X.shape[0] // batch_size) + 1
    features = []
    for n in range(N_batch):
        batch = X[n * batch_size:(n + 1) * batch_size].to(device)
        features.append(spectr(batch))
    return torch.cat(features).unsqueeze(1)

MX_train = extract_features(X_train)
MX_test = extract_features(X_test)
MX_val = extract_features(X_val)

# Create DataLoaders
def create_dataloader(X, y):
    dataset = TensorDataset(X.cpu(), y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(MX_train, y_train)
test_loader = create_dataloader(MX_test, y_test)
val_loader = create_dataloader(MX_val, y_val)

print("Data preprocessing completed.")

# ============================================= Model =============================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# Dynamically determine the number of classes from the dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_num_classes(csv_file):
    df = pd.read_csv(csv_file)
    encoder = LabelEncoder()
    encoder.fit(df['Class'])
    return len(encoder.classes_)

num_classes = get_num_classes("train.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate models with the correct number of classes
model_mel = ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=num_classes).to(device)
model_wst_1 = ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=num_classes).to(device)
model_wst_2 = ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=num_classes).to(device)

# Optimizers and Schedulers
optimizer_mel = torch.optim.AdamW(model_mel.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
scheduler_mel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mel, 'min')

optimizer_1 = torch.optim.AdamW(model_wst_1.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, 'min')

optimizer_2 = torch.optim.AdamW(model_wst_2.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min')

# ============================================= Training =============================================

def training_resnet(model, train_dataloader, val_dataloader, optimizer, scheduler, fname):
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100
    loss_train, acc_train, acc_eval, loss_eval = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        loss_ep_train = 0
        n_samples, n_correct = 0, 0

        for i, (x, labels) in enumerate(train_dataloader):
            x, labels = x.to(device), labels.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            loss_ep_train += loss.item()
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

            # Print progress
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        # Compute training accuracy
        acc_train.append(100 * n_correct / n_samples)
        loss_train.append(loss_ep_train / len(train_dataloader))

        # Validation phase
        model.eval()
        loss_ep_eval, n_samples, n_correct = 0, 0, 0

        with torch.no_grad():
            for x, labels in val_dataloader:
                x, labels = x.to(device), labels.to(device, dtype=torch.long)
                outputs = model(x)
                loss_ep_eval += criterion(outputs, labels).item()
                _, predictions = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predictions == labels).sum().item()

        # Compute validation accuracy
        acc_eval.append(100 * n_correct / n_samples)
        loss_eval.append(loss_ep_eval / len(val_dataloader))
        print(f'Validation Accuracy: {acc_eval[-1]:.2f}%')

        # Scheduler step
        scheduler.step(loss_eval[-1])

    # Save results
    res = np.array([loss_train, loss_eval, acc_train, acc_eval])
    np.save(f'{fname}_{batch_size}.npy', res)

# Train models
models = [
    (model_mel, train_loader, val_loader, optimizer_mel, scheduler_mel, 'modelmel'),
    (model_wst_1, train_loader, val_loader, optimizer_1, scheduler_1, 'modelws1'),
    (model_wst_2, train_loader, val_loader, optimizer_2, scheduler_2, 'modelws2')
]

for model, train_loader, val_loader, optimizer, scheduler, fname in models:
    training_resnet(model, train_loader, val_loader, optimizer, scheduler, fname)

# ============================================= Feature Extraction for Final MLP =============================================

# Extract probabilities for the final MLP
train_loader_1_fin = DataLoader(TensorDataset(MX_train, y_train), batch_size=batch_size, shuffle=False)
val_loader_1_fin = DataLoader(TensorDataset(MX_val, y_val), batch_size=batch_size, shuffle=False)

list_prob_1, list_prob_2 = [], []
list_prob_1_val, list_prob_2_val = [], []

with torch.no_grad():
    for x, _ in train_loader_1_fin:
        x = x.to(device)
        list_prob_1.append(model_wst_1(x))
    
    for x, _ in val_loader_1_fin:
        x = x.to(device)
        list_prob_1_val.append(model_wst_1(x))

    for x, _ in train_loader_1_fin:
        x = x.to(device)
        list_prob_2.append(model_wst_2(x))

    for x, _ in val_loader_1_fin:
        x = x.to(device)
        list_prob_2_val.append(model_wst_2(x))

# Concatenate extracted features
prob_train_1, prob_train_2 = torch.cat(list_prob_1), torch.cat(list_prob_2)
prob_val_1, prob_val_2 = torch.cat(list_prob_1_val), torch.cat(list_prob_2_val)

train_features = torch.hstack((prob_train_1, prob_train_2))
val_features = torch.hstack((prob_val_1, prob_val_2))

# Final dataset for MLP
train_final = TensorDataset(train_features, y_train)
val_final = TensorDataset(val_features, y_val)

train_final_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True)
val_final_loader = DataLoader(val_final, batch_size=batch_size, shuffle=False)

print("Final MLP dataset created.")

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size=64, hidden1=256, hidden2=128, output_size=32):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden1)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        out = self.activation1(self.linear1(x))
        out = self.activation2(self.linear2(out))
        out = self.linear3(out)
        return out

# Initialize MLP model
model_MLP = MLP(input_size=train_features.shape[1], output_size=len(torch.unique(y_train))).to(device)

# Define training components
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Training MLP
num_epochs = 500
loss_train, acc_train, acc_eval, loss_eval = [], [], [], []

for epoch in range(num_epochs):
    model_MLP.train()
    loss_ep_train, n_samples, n_correct = 0, 0, 0

    for i, (x, labels) in enumerate(train_final_loader):
        x, labels = x.to(device), labels.to(device, dtype=torch.long)

        # Forward pass
        outputs = model_MLP(x)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        loss_ep_train += loss.item()
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        # Print progress
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_final_loader)}], Loss: {loss.item():.4f}')

    # Compute training accuracy
    acc_train.append(100 * n_correct / n_samples)
    loss_train.append(loss_ep_train / len(train_final_loader))

    # Validation phase
    model_MLP.eval()
    loss_ep_eval, n_samples, n_correct = 0, 0, 0

    with torch.no_grad():
        for x, labels in val_final_loader:
            x, labels = x.to(device), labels.to(device, dtype=torch.long)
            outputs = model_MLP(x)
            loss_ep_eval += criterion(outputs, labels).item()
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

    # Compute validation accuracy
    acc_eval.append(100 * n_correct / n_samples)
    loss_eval.append(loss_ep_eval / len(val_final_loader))

    # Adjust learning rate based on validation loss
    scheduler.step(loss_eval[-1])

    # Print validation accuracy periodically
    if epoch % 50 == 0:
        print(f'Validation Accuracy after {epoch+1} epochs: {acc_eval[-1]:.2f}%')

# Save results
results = np.array([loss_train, loss_eval, acc_train, acc_eval])
np.save(f'MLP_results_{batch_size}.npy', results)

print("MLP training complete!")

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

num_epochs = 500
loss_train, acc_train, acc_eval, loss_eval = [], [], [], []

for epoch in range(num_epochs):
    model_MLP.train()
    loss_ep_train, n_samples, n_correct = 0, 0, 0

    for i, (x, labels) in enumerate(train_final_loader):  # Updated variable name
        x, labels = x.to(device), labels.to(device, dtype=torch.long)

        # Forward pass
        outputs = model_MLP(x)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        loss_ep_train += loss.item()
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        # Print progress every 100 steps
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_final_loader)}], Loss: {loss.item():.4f}')

    # Compute training accuracy
    acc_train.append(100 * n_correct / n_samples)
    loss_train.append(loss_ep_train / len(train_final_loader))

    # Validation phase
    model_MLP.eval()
    loss_ep_eval, n_samples, n_correct = 0, 0, 0

    with torch.no_grad():
        for x, labels in val_final_loader:  # Updated variable name
            x, labels = x.to(device), labels.to(device, dtype=torch.long)
            outputs = model_MLP(x)
            loss_ep_eval += criterion(outputs, labels).item()
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

    # Compute validation accuracy
    acc_eval.append(100 * n_correct / n_samples)
    loss_eval.append(loss_ep_eval / len(val_final_loader))

    # Adjust learning rate based on validation loss
    scheduler.step(loss_eval[-1])

    # Print validation accuracy every 100 epochs
    if epoch % 100 == 0:
        print(f'Validation Accuracy after {epoch+1} epochs: {acc_eval[-1]:.2f}%')

# Save training results
results = np.array([loss_train, loss_eval, acc_train, acc_eval])

namefile = f'MLP_S+Mel{batch_size}'  # Removed {J, Q} as they were undefined
np.save(namefile, results)

print("MLP training complete!")
