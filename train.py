import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import FER2013Dataset
from model import EmotionCNN

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

full_dataset_train = FER2013Dataset(csv_file='fer2013.csv', transform=train_transforms)
full_dataset_val = FER2013Dataset(csv_file='fer2013.csv', transform=val_transforms)

total_samples = len(full_dataset_train)
train_size = int(0.85 * total_samples)

indices = torch.randperm(total_samples).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

model = EmotionCNN(num_classes=7).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

for epoch in range(EPOCHS):
  model.train()
  running_loss = 0.0
  correct_train = 0
  total_train = 0

  for images, labels in train_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total_train += labels.size(0)
    correct_train += (predicted == labels).sum().item()

  epoch_loss = running_loss / len(train_loader.dataset)
  epoch_acc = (correct_train / total_train) * 100

  model.eval()
  val_loss = 0.0
  correct_val = 0
  total_val = 0

  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(DEVICE), labels.to(DEVICE)
      outputs = model(images)
      loss = criterion(outputs, labels)

      val_loss += loss.item() * images.size(0)
      _, predicted = torch.max(outputs.data, 1)
      total_val += labels.size(0)
      correct_val += (predicted == labels).sum().item()

  epoch_val_loss = val_loss / len(val_loader.dataset)
  epoch_val_acc = (correct_val / total_val) * 100

  print(
      f'Epoch [{epoch+1}/{EPOCHS}] -> '
      f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | '
      f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%'
  )
  scheduler.step(epoch_val_loss)

torch.save(model.state_dict(), 'emotion_cnn.pth')
print('Model successfully saved to emotion_cnn.pth!')