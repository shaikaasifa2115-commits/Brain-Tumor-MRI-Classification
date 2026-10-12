# Brain Tumor MRI Classification Using Deep Learning

## Project Overview
This project uses deep learning to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

The project compares a custom Convolutional Neural Network (CNN) with a MobileNetV2 transfer-learning model.

## Objectives
- Classify brain MRI images into four categories.
- Explore and visualize the dataset.
- Preprocess MRI images for model training.
- Train and evaluate CNN and MobileNetV2 models.
- Display predictions through a Python application.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter Notebook

## Dataset
The dataset contains 2,443 MRI images divided into training, validation, and testing sets.

The four classes are:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The dataset is not included in this repository.

## Models
- Custom CNN
- MobileNetV2 using transfer learning

## Evaluation
The models are evaluated using accuracy, loss curves, confusion matrices, and classification reports.

## Installation

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Download the dataset separately and arrange it according to the folder structure expected by the training scripts.

## Running the Project

To train the MobileNetV2 model:

`python train_mobilenet.py`

To run the prediction application:

`python predict.py`

Check the scripts for any required dataset paths or model filenames before running them.

## Disclaimer
This project is for educational and research purposes only. Its predictions must not be used as a substitute for professional medical diagnosis.