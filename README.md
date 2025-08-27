# COVID-19 Chest X-Ray Classification Project

## Overview
This project implements a deep learning model to classify chest X-ray images as COVID-19 positive or negative using transfer learning with PyTorch.

## Objective
Train a binary classification model that can distinguish between COVID-19 positive and normal chest X-ray images with >50% accuracy.

## Dataset Options

### 1. Primary Dataset (Recommended)
- **COVID-19 Radiography Database** from Kaggle
- Link: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Contains COVID-19, Normal, and Pneumonia chest X-ray images
- Well-balanced dataset with good image quality

### 2. Alternative Datasets
- **IEEE8023 COVID Chest X-ray Dataset**: https://github.com/ieee8023/covid-chestxray-dataset
- **DeepCOVID Dataset**: https://github.com/shervinmin/DeepCovid.git (from research paper)
- **V7 Labs COVID-19 Dataset**: https://github.com/v7labs/covid-19-xray-dataset

## Model Architecture
- **Base Model**: ResNet-18 with ImageNet pre-trained weights
- **Transfer Learning**: Fine-tuning last layers while freezing early features
- **Classification Head**: Custom fully connected layers with dropout
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (COVID vs Normal)

## Quick Start

### 1. Setup Environment
```bash
# Clone or download this project
cd "PP7: Computer vision and image classification"

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Option 1: Using Kaggle API
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip

# Option 2: Manual download from Kaggle website
# Extract COVID and Normal folders to data/ directory
```

### 3. Run the Notebook
```bash
jupyter notebook covid_classification.ipynb
```

## Project Structure
```
PP7: Computer vision and image classification/
├── covid_classification.ipynb    # Main implementation notebook
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
├── data/                        # Dataset directory
│   ├── COVID/                   # COVID-19 positive X-rays
│   └── Normal/                  # Normal chest X-rays
└── models/                      # Saved model weights
    └── covid_classifier.pth     # Trained model
```

## Key Features
- **Transfer Learning**: Leverages pre-trained ResNet for medical image analysis
- **Data Augmentation**: Improves model robustness with rotation and flipping
- **Comprehensive Evaluation**: Includes accuracy, sensitivity, specificity, and confusion matrix
- **Visualization**: Training curves and performance metrics
- **Reproducible**: Fixed random seeds for consistent results

## Expected Results
- **Target Accuracy**: >50% (achievable goal)
- **Typical Performance**: 80-95% accuracy with proper training
- **Key Metrics**: Sensitivity and specificity for medical applications

## Implementation Notes
- Uses PyTorch framework with torchvision models
- Implements early stopping and learning rate scheduling
- Includes proper train/test split with stratification
- Medical AI ethics and limitations discussed in reflection section

## Dataset Download Instructions

### Method 1: Kaggle API (Recommended)
1. Install Kaggle: `pip install kaggle`
2. Set up Kaggle API credentials
3. Download: `kaggle datasets download -d tawsifurrahman/covid19-radiography-database`

### Method 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. Click "Download" button
3. Extract to project directory

### Method 3: Alternative Sources
- GitHub repositories listed above
- Research paper datasets
- Medical image databases (with proper permissions)

## Important Notes
- This is for educational/research purposes only
- Not intended for clinical diagnosis
- Requires medical validation for real-world use
- Consider data privacy and ethical guidelines

## Next Steps
1. Download and prepare dataset
2. Run the Jupyter notebook
3. Experiment with different architectures
4. Upload to GitHub for sharing and review