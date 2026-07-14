import torch.optim as optim
import torch.nn as nn

class EmotionCNN(nn.Module):
  def __init__(self, num_classes=7):
    super().__init__()
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    
    self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.bn1_1 = nn.BatchNorm2d(32)
    self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.bn1_2 = nn.BatchNorm2d(32)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.drop1 = nn.Dropout(0.2)

    self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.bn2_1 = nn.BatchNorm2d(64)
    self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.bn2_2 = nn.BatchNorm2d(64)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.drop2 = nn.Dropout(0.25)

    self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn3_1 = nn.BatchNorm2d(128)
    self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.bn3_2 = nn.BatchNorm2d(128)
    self.pool3 = nn.MaxPool2d(2, 2)
    self.drop3 = nn.Dropout(0.3)

    self.fc1 = nn.Linear(128 * 6 * 6, 512) 
    self.bn_fc = nn.BatchNorm1d(512)
    self.drop_fc = nn.Dropout(0.5)
    
    self.fc2 = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.relu(self.bn1_1(self.conv1_1(x)))
    x = self.pool1(self.relu(self.bn1_2(self.conv1_2(x))))
    x = self.drop1(x)

    x = self.relu(self.bn2_1(self.conv2_1(x)))
    x = self.pool2(self.relu(self.bn2_2(self.conv2_2(x))))
    x = self.drop2(x)

    x = self.relu(self.bn3_1(self.conv3_1(x)))
    x = self.pool3(self.relu(self.bn3_2(self.conv3_2(x))))
    x = self.drop3(x)

    x = self.flatten(x)
    x = self.drop_fc(self.relu(self.bn_fc(self.fc1(x))))
    x = self.fc2(x)
    return x