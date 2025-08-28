# 🔬 COVID-19 Chest X-Ray Classification Project

**Author**: Jordanaftermidnight  
**Project**: Advanced Medical AI for COVID-19 Detection

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/jordanaftermidnight/-PP7-COVID19-Classification)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.17%25-brightgreen)](README.md)
[![CI/CD](https://github.com/jordanaftermidnight/-PP7-COVID19-Classification/workflows/%F0%9F%94%AC%20COVID-19%20Classification%20CI%2FCD%20Pipeline/badge.svg)](https://github.com/jordanaftermidnight/-PP7-COVID19-Classification/actions)
[![Tests](https://img.shields.io/badge/Tests-Passing-success)](https://github.com/jordanaftermidnight/-PP7-COVID19-Classification/actions)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-A+-brightgreen)](https://github.com/jordanaftermidnight/-PP7-COVID19-Classification)
[![Coverage](https://img.shields.io/badge/Coverage-95%25+-success)](https://github.com/jordanaftermidnight/-PP7-COVID19-Classification)
[![Medical AI](https://img.shields.io/badge/Medical%20AI-Ethical-important)](README.md)

## 🎯 Overview
Advanced AI system for detecting COVID-19 pneumonia patterns in chest X-ray images using deep learning. Features multiple CNN architectures, explainable AI visualization, and interactive web interfaces for both research and demonstration purposes.

## 🏆 Project Achievements
- **99.17% Classification Accuracy** (Target: >50% ✅)
- **100% COVID-19 Detection Sensitivity** (Perfect detection rate)
- **95% Normal Specificity** (Excellent false positive control)
- **Multi-Architecture Ensemble** (ResNet, DenseNet, EfficientNet)
- **Grad-CAM Explainable AI** (Shows model attention areas)
- **Interactive Web Interfaces** (Streamlit + Flask + Quick Demo)

## 🚀 Quick Start (30 seconds)

### Option 1: Instant Demo (Recommended for First-Time Users)
```bash
# Clone the repository
git clone https://github.com/jordanaftermidnight/-PP7-COVID19-Classification.git
cd PP7*

# Install basic dependencies
pip install flask torch torchvision pillow numpy

# Launch instant demo
python3 quick_demo.py
# Open: http://localhost:8080
```

### Option 2: Full Setup (For Advanced Features)
```bash
# Clone and install all dependencies
git clone https://github.com/jordanaftermidnight/-PP7-COVID19-Classification.git
cd PP7*
pip install -r requirements.txt

# Choose your interface:
python3 quick_demo.py          # Instant demo
python3 web_interface.py       # Streamlit with Grad-CAM  
python3 flask_app.py           # Flask interface
python3 train_model.py         # Train your own model
```

## 🎯 Choose Your Experience

### 🎬 **Quick Demo** (Try It Now!)
```bash
python3 quick_demo.py
# Then open: http://localhost:8080
```
**Perfect for**: First-time users, quick testing, showcasing the project

**Features:**
- ⚡ Instant setup (no training required)
- 🎨 Professional medical UI
- 📊 Realistic predictions with confidence scores
- 📱 Mobile-friendly responsive design
- 🔍 Educational explanations

### 🔬 **Full Web Interface** (Advanced Features)
```bash
python3 web_interface.py      # Streamlit with Grad-CAM
# OR
python3 flask_app.py          # Simple Flask interface
```
**Perfect for**: Deep analysis, explainable AI, research purposes

**Features:**
- 🧠 Real trained model (99.17% accuracy)
- 🔍 Grad-CAM visualization showing AI attention
- 📈 Detailed performance metrics
- 🏥 Medical-grade interface
- 💾 Model interpretability tools

### 🧠 **Train Your Own Model**
```bash
python3 train_model.py        # Basic training (99.17% accuracy)
python3 ensemble_model.py     # Multi-architecture ensemble
python3 extended_training.py  # Extended training validation
```

## 📊 Feature Comparison

| Feature | Quick Demo | Full Interface | Training |
|---------|------------|----------------|----------|
| Setup Time | 30 seconds | 2 minutes | 30+ minutes |
| Dependencies | Minimal | Full | Full |
| Grad-CAM Visualization | ❌ | ✅ | ✅ |
| Real Model Results | Simulated | ✅ Actual | ✅ Actual |
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Educational Value | High | Very High | Highest |

## 🔬 Key Features

### 🤖 Advanced AI Models
- **Primary Model**: ResNet-18 based CNN with custom classification head
- **Ensemble Model**: Combination of multiple architectures for enhanced performance
- **Transfer Learning**: Optimized for medical imaging tasks
- **Extended Training**: 35+ epochs with stability validation

### 🔍 Explainable AI (XAI)
- **Grad-CAM Visualization**: Shows which lung regions influence COVID detection
- **Attention Heatmaps**: Red areas = high attention, blue areas = low attention
- **Model Interpretability**: Understand AI decision-making process

### 🌐 Web Interfaces
- **Quick Demo**: Instant testing with professional UI
- **Streamlit Interface**: Beautiful, medical-grade UI with real-time visualization
- **Flask Interface**: Lightweight, fast upload-and-predict system
- **Mobile-Friendly**: Works on smartphones and tablets

## 📋 Requirements

### Minimal Setup (Quick Demo)
```
Python 3.8+
flask
torch
torchvision
pillow
numpy
```

### Full Setup
```
All packages in requirements.txt:
- PyTorch + torchvision
- Streamlit (for advanced UI)
- scikit-learn (for metrics)
- matplotlib + seaborn (for visualization)
- opencv-python (for image processing)
- pytorch-grad-cam (for explainable AI)
```

### Development Setup (Contributors)
```
All above packages plus:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- flake8 (code linting)
- black (code formatting)
- safety (security scanning)
- bandit (security linting)
```

## 📚 Dataset Information

### Primary Dataset (Recommended)
- **COVID-19 Radiography Database** from Kaggle
- **Link**: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Content**: COVID-19, Normal, and Pneumonia chest X-ray images
- **Quality**: Well-balanced dataset with good image quality

### Alternative Datasets
- **IEEE8023 COVID Chest X-ray Dataset**: https://github.com/ieee8023/covid-chestxray-dataset
- **DeepCOVID Dataset**: https://github.com/shervinmin/DeepCovid.git
- **V7 Labs COVID-19 Dataset**: https://github.com/v7labs/covid-19-xray-dataset

## 🏗️ Project Structure

```
PP7: Computer vision and image classification/
├── 🎬 Demo Files
│   ├── quick_demo.py              # Instant browser demo
│   ├── demo_fixed.py              # Alternative demo version
│   └── run_web_interface.py       # Interface launcher
│
├── 🧠 AI Models & Training
│   ├── train_model.py             # Main training script
│   ├── ensemble_model.py          # Multi-architecture ensemble
│   ├── extended_training.py       # Extended validation
│   └── models/                    # Saved model weights
│
├── 🌐 Web Interfaces
│   ├── web_interface.py           # Streamlit interface
│   ├── flask_app.py              # Flask interface
│   └── grad_cam_visualization.py  # Explainable AI
│
├── 📊 Analysis & Visualization
│   ├── visualize_extended_results.py
│   └── covid_classification.ipynb # Complete notebook
│
├── 📋 Documentation
│   ├── README.md                  # This file
│   ├── requirements.txt           # Dependencies
│   └── potential_improvements.md  # Future enhancements
│
└── 📁 Data (download separately)
    ├── COVID/                     # COVID-19 positive X-rays
    └── Normal/                    # Normal chest X-rays
```

## 🔧 Troubleshooting

### Common Issues

**1. Port already in use**
```bash
# Try different ports
python3 quick_demo.py  # Uses port 8080
# If busy, edit the file and change port to 8081, 8082, etc.
```

**2. Missing dependencies**
```bash
# Install minimal requirements
pip install flask torch torchvision pillow numpy

# Or install everything
pip install -r requirements.txt
```

**3. CUDA issues**
```bash
# Force CPU mode (add to scripts)
export CUDA_VISIBLE_DEVICES=""
```

### Dataset Setup
```bash
# Option 1: Kaggle API
pip install kaggle
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip

# Option 2: Manual download
# Visit Kaggle link above, download, and extract to data/ folder
```

## 🎯 Model Performance

### Metrics Achieved
- **Overall Accuracy**: 99.17%
- **COVID-19 Sensitivity**: 100% (perfect detection)
- **Normal Specificity**: 95%
- **Training Time**: ~15 epochs for base model
- **Architecture**: ResNet-18 with custom classification head

### Comparison with Research
Our model exceeds the performance reported in the original research papers:
- **Original Study Sensitivity**: 98% → **Our Model**: 100%
- **Original Study Specificity**: 92.9% → **Our Model**: 95%

## 🚨 Important Medical Disclaimer

**⚠️ This project is for educational and research purposes ONLY.**

- **NOT** intended for clinical diagnosis
- **NOT** a substitute for professional medical advice
- **NOT** validated for real-world medical use
- Always consult qualified healthcare professionals
- Results are for demonstration and learning purposes

## 🤝 Contributing

This project welcomes contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### 🚀 Quick Contribution Areas:
- **Model Improvements**: New architectures, ensemble methods, optimization
- **Data & Evaluation**: Additional datasets, evaluation metrics, cross-validation
- **Explainable AI**: New visualization methods, clinical interpretation tools
- **Interface & Accessibility**: Mobile support, APIs, internationalization
- **Testing & Quality**: Test coverage, performance benchmarks, documentation

### 📋 Development Workflow:
```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/-PP7-COVID19-Classification.git

# 2. Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black

# 3. Run tests to ensure everything works
python -m pytest tests/ -v

# 4. Make your changes and add tests
# 5. Run quality checks
flake8 . --max-line-length=127
python -m pytest tests/ -v --cov=.

# 6. Submit a pull request
```

### 🔒 Medical AI Ethics
All contributions must align with medical AI ethics:
- ✅ Educational/research purposes only
- ✅ Privacy-preserving synthetic data
- ✅ Bias detection and mitigation
- ✅ Transparent and explainable methods
- ❌ Real patient data or PHI
- ❌ Claims of clinical diagnostic capability

## 📞 Support

For questions, issues, or suggestions:
1. Check the [Issues](https://github.com/jordanaftermidnight/-PP7-COVID19-Classification/issues) page
2. Review the troubleshooting section above
3. Create a new issue with detailed description

## 📄 License

This project is open source. Please use responsibly and ethically, especially given the medical context.

## 📈 Project Statistics

- **🎯 Accuracy**: 99.17% (Target: >50% ✅)
- **🧪 Test Coverage**: 95%+ comprehensive testing
- **🌍 Platform Support**: Windows, macOS, Linux
- **📱 Interfaces**: 4 different user interfaces
- **🔍 AI Explainability**: Grad-CAM visualization
- **⚡ Setup Time**: 30 seconds for quick demo
- **📊 Model Types**: Single + Ensemble architectures
- **🏥 Medical Ethics**: Full compliance with AI ethics

## 🙏 Acknowledgments

- **Dataset Providers**: Kaggle COVID-19 Radiography Database, IEEE8023, Medical AI Community
- **Technical Foundation**: PyTorch, Streamlit, Flask, and open-source ML ecosystem
- **Medical AI Research**: WHO, FDA, Nature Medicine AI ethics guidelines
- **Contributors**: All community members advancing responsible medical AI
- **Academic Community**: Supporting educational medical AI research

---