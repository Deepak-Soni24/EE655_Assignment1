import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Custom activation function: x * sigmoid(x)
class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Modified LeNet Architecture
class ModifiedLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)  # 3x3 filter
        self.act1 = SwishActivation()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # 3x3 filter
        self.act2 = SwishActivation()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling
        
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.act3 = SwishActivation()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = SwishActivation()
        self.fc3 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Softmax layer

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.softmax(self.fc3(x)) #Softmax
        return x

# Load Dataset
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure images are grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ImageFolder(root='MNIST Dataset JPG format/MNIST - JPG - training', transform=transform)
test_dataset = ImageFolder(root='MNIST Dataset JPG format/MNIST - JPG - testing', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

model = ModifiedLeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)


num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    scheduler.step()  # Adjust learning rate
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Testing the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
