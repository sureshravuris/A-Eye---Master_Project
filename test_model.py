import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

# Import the model architecture from your training script
from model_training import create_model

# Define class names
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
              'Age-related Macular Degeneration', 'Hypertension', 
              'Pathological Myopia', 'Other diseases/abnormalities']

# Shorthand for class names
class_shorthand = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

def load_model(model_path, num_classes=8):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess an image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def predict(model, image_tensor, device):
    """Make a prediction with the model"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = outputs.cpu().numpy()[0]
    
    # Get indices of top predictions
    threshold = 0.5
    indices = np.where(probabilities >= threshold)[0]
    
    # If no prediction meets the threshold, get the highest one
    if len(indices) == 0:
        indices = [np.argmax(probabilities)]
    
    return probabilities, indices

def display_results(image, probabilities, indices):
    """Display the image and prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display the image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Display the probabilities as a bar chart
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probabilities, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Prediction Probabilities')
    ax2.set_xlim(0, 1)
    
    # Highlight predictions above threshold
    for i in indices:
        ax2.get_yticklabels()[i].set_color('red')
        ax2.get_yticklabels()[i].set_weight('bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print results to console
    print("\nPrediction Results:")
    print("-" * 50)
    for i, prob in enumerate(probabilities):
        highlight = " <- Detected" if i in indices else ""
        print(f"{class_names[i]} ({class_shorthand[i]}): {prob:.4f}{highlight}")
    
    print("\nDetected Conditions:")
    for i in indices:
        print(f"- {class_names[i]} with {probabilities[i]:.4f} confidence")

def main():
    parser = argparse.ArgumentParser(description='Test eye disease detection model on a single image')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to the model')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    # Load model
    model, device = load_model(args.model)
    print(f"Model loaded from {args.model} using {device}")
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(args.image)
    
    # Make prediction
    probabilities, indices = predict(model, image_tensor, device)
    
    # Display results
    display_results(original_image, probabilities, indices)

if __name__ == "__main__":
    main()