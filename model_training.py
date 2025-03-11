# model_training.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 224  # ResNet expects 224x224 images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define the dataset class
class EyeDiseaseDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, use_right_eye=True):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.use_right_eye = use_right_eye
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.dataframe.iloc[idx]
        
        # Select which eye to use
        if self.use_right_eye:
            img_name = row['Right-Fundus']
        else:
            img_name = row['Left-Fundus']
        
        # First try the preprocessed_images folder
        img_path = os.path.join('data/preprocessed_images', img_name)
        
        # If not found, try the training/testing folders
        if not os.path.exists(img_path):
            img_path_training = os.path.join('data/ODIR-5K/ODIR-5K/Training Images', img_name)
            img_path_testing = os.path.join('data/ODIR-5K/ODIR-5K/Testing Images', img_name)
            
            if os.path.exists(img_path_training):
                img_path = img_path_training
            elif os.path.exists(img_path_testing):
                img_path = img_path_testing
        
        # Load and transform the image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if the file can't be loaded
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))
        
        # Get the target labels
        target = torch.tensor(eval(row['target']), dtype=torch.float32)
        
        return {'image': image, 'target': target}

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create a model
def create_model(num_classes=8):
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    
    # Freeze the early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()  # Sigmoid for multi-label classification
    )
    
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs = batch['image'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = batch['image'].to(DEVICE)
                targets = batch['target'].to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
        
        print()
    
    return model, history

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch['image'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # Binary prediction threshold
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate accuracy for each class
    correct_per_class = (all_preds == all_targets).sum(axis=0)
    total_per_class = all_targets.shape[0]
    accuracy_per_class = correct_per_class / total_per_class
    
    # Calculate overall accuracy
    overall_accuracy = (all_preds == all_targets).mean()
    
    return {
        'accuracy_per_class': accuracy_per_class,
        'overall_accuracy': overall_accuracy
    }

if __name__ == "__main__":
    # Load the data
    data_file = 'data/full_df.csv'
    df = pd.read_csv(data_file)
    
    # Check if we have images in the preprocessed_images folder
    preprocessed_dir = 'data/preprocessed_images'
    training_dir = 'data/ODIR-5K/ODIR-5K/Training Images'
    testing_dir = 'data/ODIR-5K/ODIR-5K/Testing Images'
    
    # Check which directories exist
    dirs_exist = {
        'preprocessed': os.path.exists(preprocessed_dir),
        'training': os.path.exists(training_dir),
        'testing': os.path.exists(testing_dir)
    }
    
    print("Available image directories:")
    for dir_name, exists in dirs_exist.items():
        print(f"{dir_name}: {'Yes' if exists else 'No'}")
    
    # Verify some image files exist in the dataframe
    sample_right_eye = df['Right-Fundus'].iloc[0]
    sample_left_eye = df['Left-Fundus'].iloc[0]
    
    print(f"\nSample image names from dataframe:")
    print(f"Right eye: {sample_right_eye}")
    print(f"Left eye: {sample_left_eye}")
    
    # Check if these files exist in our directories
    for dir_path in [preprocessed_dir, training_dir, testing_dir]:
        if os.path.exists(dir_path):
            right_path = os.path.join(dir_path, sample_right_eye)
            left_path = os.path.join(dir_path, sample_left_eye)
            print(f"\nChecking in {dir_path}:")
            print(f"Right eye exists: {os.path.exists(right_path)}")
            print(f"Left eye exists: {os.path.exists(left_path)}")
    
    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # We'll use the directory where we found the images
    # Create datasets - we don't specify img_dir because our dataset class checks multiple directories
    train_dataset = EyeDiseaseDataset(train_df, '', transform=train_transform)
    val_dataset = EyeDiseaseDataset(val_df, '', transform=val_transform)
    test_dataset = EyeDiseaseDataset(test_df, '', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create and train the model
    model = create_model().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on the test set
    results = evaluate_model(model, test_loader)
    
    print("\nTest Results:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print("\nAccuracy per class:")
    class_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    for i, cls in enumerate(class_names):
        print(f"{cls}: {results['accuracy_per_class'][i]:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()