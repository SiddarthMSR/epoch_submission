import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

# For loading/preprocessing our data (took help of ChatGPT for code here)
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row.iloc[0]
        try:
            image_data = row.iloc[1:].values.astype(np.float32).reshape((28, 28))
            if np.any(np.isnan(image_data)):  # Check for NaN values
                return None
            image = Image.fromarray(image_data.astype(np.uint8))

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            #For handling (purposfully) corrupted rows.
            print(f"Error processing row {idx}: {e}")
            return None


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_data = CSVDataset(csv_file='alphabets_28x28.csv', transform=transform)


def collate_fn(batch):
    batch = [data for data in batch if data is not None]   #Just removing the empty data.
    return torch.utils.data.dataloader.default_collate(batch)  

# Creating data loader for training
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

#CNN architecture (took help of GPT to adjust parameters)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust based on image size after pooling
        self.fc2 = nn.Linear(128, 26)  # 26 classes (A-Z)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 3
for epoch in range(epochs):
    cur_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        if data is None:
            continue  

        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = torch.tensor([ord(label) - ord('A') for label in labels], dtype=torch.long)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        cur_loss += loss.item()

        if i % 100 == 99:  
            print(f"[{epoch + 1}, {i + 1}] loss: {cur_loss / 100:.3f}")
            cur_loss = 0.0

print("Finished Training")

# Saving the model
torch.save(model.state_dict(), 'alphabet_recognition_model.pth')
