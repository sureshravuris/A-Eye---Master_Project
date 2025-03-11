# A-Eye: Master Project - Eye Disease Detection

This project focuses on detecting eye diseases using deep learning. The model has been trained on medical images and can be used to classify and analyze test images.

## 📂 Project Structure
```
|-- data/                # Place your dataset here (not included in the repo)
|-- test_images/         # Sample test images for inference
|-- model_training.py    # Script to train the model
|-- data_exploration.py  # Script for exploring dataset
|-- test_model.py        # Script to test the trained model
|-- best_model.pth       # Pre-trained model file
|-- training_history.png # Training history visualization
|-- sample_image.png     # Example image for testing
|-- README.md            # This file
```

---
## 🚀 How to Run This Project

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/sureshravuris/A-Eye---Master_Project.git
cd eye-disease-detection
```

### 2️⃣ Set Up the Virtual Environment
```sh
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Download the Dataset
The dataset is not included in this repository due to size constraints. You can download it from the following link:
[Dataset Link](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

After downloading, place the dataset inside the `data/` folder.

### 5️⃣ Train the Model
```sh
python model_training.py
```
This will train the model and save the best-performing model as `best_model.pth`.

### 6️⃣ Test the Model
To run inference on a test image:
```sh
python test_model.py --image test_images/sample_image.png
```
This will output the predicted result.

### 7️⃣ Model Evaluation
To visualize the training process:
```sh
python data_exploration.py
```
This script generates insights and statistics about the dataset.
