# train.py
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
MANUAL_SEED = 42
BATCH_SIZE = 32
SHUFFLE = True
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset Class
class LandscapeDataset(Dataset):
    def __init__(self, transform=None):
        self.dataroot = 'training_images\landscape Images'  # Update this path to your dataset location
        self.images = os.listdir(f'{self.dataroot}/color')
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        color_img = read_image(f'{self.dataroot}/color/{img_path}') / 255
        gray_img = read_image(f'{self.dataroot}/gray/{img_path}') / 255

        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)

        return color_img, gray_img

# Model Architecture
class ColorAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 3, stride=2)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, 3, 3, stride=2, output_padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))
        
        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(torch.cat((u1,d3), dim=1)))
        u3 = self.relu(self.up3(torch.cat((u2, d2), dim=1)))
        u4 = self.sigmoid(self.up4(torch.cat((u3,d1), dim=1)))
        
        return u4

def train_model():
    # Set random seed
    torch.manual_seed(MANUAL_SEED)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((150, 150), antialias=True),
    ])
    
    # Load dataset
    print("Loading dataset...")
    dataset = LandscapeDataset(transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size], 
                                     generator=torch.Generator().manual_seed(MANUAL_SEED))
    
    # Create dataloaders
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    
    print(f"Training on {len(train_set)} images, Testing on {len(test_set)} images")
    
    # Initialize model
    model = ColorAutoEncoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader))
        
        for idx, (color_img, gray_img) in progress_bar:
            color_img = color_img.to(DEVICE)
            gray_img = gray_img.to(DEVICE)
            
            # Forward pass
            predictions = model(gray_img)
            loss = criterion(predictions, color_img)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}')
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "colorization_model.pth")
            print(f"Saved new best model with loss: {best_loss:.6f}")
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for color_img, gray_img in tqdm(testloader, desc="Testing"):
            color_img = color_img.to(DEVICE)
            gray_img = gray_img.to(DEVICE)
            predictions = model(gray_img)
            loss = criterion(predictions, color_img)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(testloader)
    print(f"\nFinal test loss: {avg_test_loss:.6f}")
    print("\nTraining completed! Model saved as 'colorization_model.pth'")

if __name__ == "__main__":
    train_model()
