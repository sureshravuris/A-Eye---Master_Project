# data_exploration.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Update this path to your CSV file
data_file = 'data/full_df.csv'

# Check if the file exists
if not os.path.exists(data_file):
    print(f"Error: File '{data_file}' not found.")
    exit(1)

# Load the data
try:
    # Load the CSV file
    df = pd.read_csv(data_file)
    
    # Basic information
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    
    # Check for diagnostic columns
    diagnostic_cols = [col for col in df.columns if col in ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]
    if diagnostic_cols:
        print("\nDiagnosis distribution:")
        for col in diagnostic_cols:
            count = df[col].sum()
            percentage = (count / len(df)) * 100
            print(f"{col}: {count} ({percentage:.2f}%)")
    
    # Look for image paths
    img_cols = [col for col in df.columns if 'eye' in col.lower() or 'fundus' in col.lower() or 'image' in col.lower()]
    if img_cols:
        print("\nImage columns found:", img_cols)
        
        # Try to load a sample image
        img_dir = 'data/ODIR-5K/Training Images'
        if not os.path.exists(img_dir):
            img_dir = 'data/ODIR-5K/Images'
            if not os.path.exists(img_dir):
                # Try to find the images directory
                for root, dirs, files in os.walk('data'):
                    for dir in dirs:
                        if 'image' in dir.lower() or 'fundus' in dir.lower():
                            img_dir = os.path.join(root, dir)
                            break
        
        if os.path.exists(img_dir):
            # Get the first image name from the dataframe
            if len(img_cols) > 0 and len(df) > 0:
                sample_img_name = df.iloc[0][img_cols[0]]
                sample_img_path = os.path.join(img_dir, sample_img_name)
                
                if os.path.exists(sample_img_path):
                    print(f"\nSample image path: {sample_img_path}")
                    img = Image.open(sample_img_path)
                    print(f"Image size: {img.size}")
                    
                    # Plot the image
                    plt.figure(figsize=(8, 6))
                    plt.imshow(np.array(img))
                    plt.title(f"Sample Image: {sample_img_name}")
                    plt.axis('off')
                    plt.savefig('sample_image.png')
                    print("Sample image saved as 'sample_image.png'")
                else:
                    print(f"Warning: Sample image not found at {sample_img_path}")
                    # Try to find any image in the directory
                    image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
                    if image_files:
                        sample_img_path = os.path.join(img_dir, image_files[0])
                        print(f"Using alternative image: {sample_img_path}")
                        img = Image.open(sample_img_path)
                        print(f"Image size: {img.size}")
                        plt.figure(figsize=(8, 6))
                        plt.imshow(np.array(img))
                        plt.title(f"Sample Image: {image_files[0]}")
                        plt.axis('off')
                        plt.savefig('sample_image.png')
                        print("Sample image saved as 'sample_image.png'")
            else:
                print("No image columns or empty dataframe.")
        else:
            print(f"Warning: Image directory not found")
            # Try to find images in any subdirectory
            for root, dirs, files in os.walk('data'):
                image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
                if image_files:
                    sample_img_path = os.path.join(root, image_files[0])
                    print(f"Found image: {sample_img_path}")
                    img = Image.open(sample_img_path)
                    print(f"Image size: {img.size}")
                    break
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    # Display data types
    print("\nData types:")
    print(df.dtypes)
    
except Exception as e:
    print(f"Error exploring data: {e}")