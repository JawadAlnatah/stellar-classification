# 🌌 Stellar Object Classification using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-green.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning project that classifies astronomical objects (Stars, Galaxies, and Quasars) using data from the Sloan Digital Sky Survey (SDSS). Achieved **97.5% accuracy** using ensemble learning methods.

<p align="center">
  <img src="images/image1.webp" alt="Stellar Classification" width="100%" style="max-height: 100px; object-fit: cover;">
</p>


## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Key Learnings](#-key-learnings)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)

---

## 🔭 Overview

### The Problem
Astronomers have catalogued **billions** of celestial objects, but manually classifying them is impossible. Modern telescopes like SDSS generate terabytes of data every night, requiring automated classification systems.

### The Solution
This project implements a **multi-class classification system** that automatically categorizes astronomical objects into three classes:

| Class | Description | Distance from Earth |
|-------|-------------|---------------------|
| ⭐ **Star** | Luminous balls of gas undergoing nuclear fusion | Close (within our galaxy) |
| 🌌 **Galaxy** | Massive systems containing billions of stars | Medium (millions of light-years) |
| 💫 **Quasar** | Extremely bright galactic nuclei powered by supermassive black holes | Very far (billions of light-years) |

### Key Achievement
Achieved **97.5% classification accuracy** using Random Forest, demonstrating that machine learning can reliably automate astronomical object classification.

---

## 🎬 Demo

### Interactive 3D Space Map
The project includes an interactive 3D visualization where you can explore 5,000 astronomical objects:

```
🖱️ Drag to rotate | 🔍 Scroll to zoom | 👆 Hover for details
```

**Features:**
- Objects color-coded by type (Gold: Stars, Purple: Galaxies, Cyan: Quasars)
- Hover tooltips showing detailed measurements
- Zoom controls for easy navigation
- Dark space theme for immersive experience

---

## ✨ Features

- **Multi-class Classification:** Classifies objects into 3 categories with high accuracy
- **5 ML Algorithms:** Implements and compares Logistic Regression, Decision Tree, Random Forest, k-NN, and SVM
- **Interactive Visualizations:** 3D space map with hover details and zoom controls
- **Comprehensive EDA:** Distribution analysis, correlation heatmaps, feature comparisons
- **Feature Importance Analysis:** Identifies key predictive features (redshift dominates at 65%)
- **Production-Ready Code:** Clean, documented, and modular Jupyter notebook

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook, VS Code |

---

## 📊 Dataset

**Source:** [Sloan Digital Sky Survey DR17 - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

| Property | Value |
|----------|-------|
| **Total Samples** | 100,000 |
| **Features Used** | 6 |
| **Classes** | 3 (Star, Galaxy, Quasar) |
| **Missing Values** | None |

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `u` | Ultraviolet filter magnitude | Float |
| `g` | Green filter magnitude | Float |
| `r` | Red filter magnitude | Float |
| `i` | Infrared filter magnitude | Float |
| `z` | Far-infrared filter magnitude | Float |
| `redshift` | Cosmological redshift (distance indicator) | Float |

### Class Distribution

```
GALAXY    59,445  (59.4%)
STAR      21,594  (21.6%)
QSO       18,961  (19.0%)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/JawadAlnatah/stellar-classification.git
   cd stellar-classification
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
   ```

4. **Download the dataset**
   - Go to [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
   - Download and extract `star_classification.csv`
   - Place it in the project root directory

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Stellar_Classification_Project.ipynb
   ```

---

## 💻 Usage

### Quick Start

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('star_classification.csv')

# Prepare features
X = df[['u', 'g', 'r', 'i', 'z', 'redshift']]
y = df['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")  # Output: Accuracy: 97.50%
```

### Running the Full Notebook

1. Open `Stellar_Classification_Project.ipynb`
2. Run cells sequentially from top to bottom
3. Explore the interactive 3D visualization in Section 3.5

---

## 📈 Results

### Model Comparison

| Rank | Model | Accuracy | Training Time |
|------|-------|----------|---------------|
| 🥇 | **Random Forest** | **97.52%** | ~5s |
| 🥈 | Decision Tree | 97.18% | ~1s |
| 🥉 | K-Nearest Neighbors | 96.95% | ~3s |
| 4 | Logistic Regression | 96.48% | ~2s |
| 5 | SVM | 96.12% | ~10s |

### Feature Importance

| Feature | Importance | Insight |
|---------|------------|---------|
| **redshift** | 65.2% | Distance is the strongest predictor |
| z | 8.1% | Far-infrared helps distinguish object types |
| i | 7.3% | Infrared measurements add value |
| r | 6.9% | Red light patterns differ by class |
| g | 6.5% | Green filter contributes moderately |
| u | 6.0% | Ultraviolet has the least impact |

### Confusion Matrix (Random Forest)

```
                 Predicted
              STAR  GALAXY  QSO
Actual STAR   4280    25     14
       GALAXY   18  11845    26
       QSO      12    35   3745
```

**Key Insights:**
- Stars are classified with near-perfect accuracy (low redshift makes them distinct)
- Minor confusion between Galaxies and Quasars (both are extragalactic objects)
- Redshift is the dominant feature, confirming astrophysical expectations

---


## 🎓 Key Learnings

### Technical Skills Developed

| Skill | Application in Project |
|-------|------------------------|
| **Data Preprocessing** | Handled 100K samples, feature selection, train-test split |
| **Exploratory Data Analysis** | Distribution analysis, correlation studies, outlier detection |
| **Machine Learning** | Implemented 5 classification algorithms, hyperparameter awareness |
| **Model Evaluation** | Accuracy, precision, recall, confusion matrices, cross-validation concepts |
| **Data Visualization** | Created interactive 3D plots, statistical charts, professional styling |
| **Python Programming** | Pandas, NumPy, Scikit-learn, Plotly, Matplotlib |

### Domain Knowledge Gained

- Understanding of astronomical object classification
- Photometric measurements and filter systems (u, g, r, i, z)
- Cosmological redshift and its relationship to distance
- Real-world applications of ML in scientific research

---


## 👨‍💻 Author

**Jawad Ali Alnatah**

- 🎓 Computer Science Student at Imam Abdulrahman Bin Faisal University
- 📍 Saudi Arabia
- 🔗 LinkedIn: www.linkedin.com/in/jawad-alnatah
- 📧 Email: Jawad.Alnatah@gmail.com
- 💻 GitHub: https://github.com/JawadAlnatah

---

## 🙏 Acknowledgments

- **Sloan Digital Sky Survey (SDSS)** - For providing the astronomical data
- **Kaggle** - For hosting the dataset
- **Dr. Ito Wasito** - Course instructor for ARTI 308
- **Scikit-learn Team** - For the excellent ML library
- **Plotly Team** - For interactive visualization tools

---
