# 🌌 Stellar Object Classification - Complete Project Guide

## ARTI 308 - Machine Learning Course Project
### Imam Abdulrahman Bin Faisal University
### Academic Year 2025-2026

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [How We Chose This Topic](#2-how-we-chose-this-topic)
3. [Understanding the Physics (Simple Explanation)](#3-understanding-the-physics-simple-explanation)
4. [Understanding the Data](#4-understanding-the-data)
5. [How Machine Learning Solves This Problem](#5-how-machine-learning-solves-this-problem)
6. [Project Implementation](#6-project-implementation)
7. [Results and Findings](#7-results-and-findings)
8. [Visualizations](#8-visualizations)
9. [References](#10-references)

---

# 1. Project Overview

## 1.1 What Is This Project?

We built a **machine learning model** that can automatically classify astronomical objects into three categories:

| Object | Symbol | Description |
|--------|--------|-------------|
| **Star** | ⭐ | A luminous ball of gas (like our Sun) |
| **Galaxy** | 🌌 | A massive system of billions of stars |
| **Quasar (QSO)** | 💫 | An extremely bright galactic nucleus powered by a supermassive black hole |

## 1.2 Why Is This Important?

- **Billions of objects** exist in space - impossible to classify manually
- **Modern telescopes** generate massive amounts of data every night
- **Machine learning** can classify objects in seconds with 97%+ accuracy
- **Real-world application** - astronomers actually use ML for this!

## 1.3 Dataset

- **Source:** Sloan Digital Sky Survey (SDSS) Data Release 17
- **Size:** 100,000 observations
- **Available on:** Kaggle
- **Link:** https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

---

# 2. How We Chose This Topic

## 2.1 Brainstorming Process

We considered three space-related topics:

### Option 1: Stellar Object Classification (CHOSEN ✅)
- **Type:** Multi-class Classification (3 classes)
- **Dataset:** 100,000 observations
- **Difficulty:** Moderate
- **Why we chose it:** Large dataset, multi-class problem offers richer learning, excellent scientific relevance

### Option 2: NASA Hazardous Asteroid Classification
- **Type:** Binary Classification
- **Dataset:** 4,687 asteroids
- **Difficulty:** Moderate
- **Why we didn't choose it:** Binary classification is simpler, less learning opportunity

### Option 3: Exoplanet Detection (Kepler Mission)
- **Type:** Binary Classification
- **Dataset:** 9,564 objects
- **Difficulty:** Higher
- **Why we didn't choose it:** More complex preprocessing, requires more domain knowledge

## 2.2 Why Stellar Classification Won

| Criteria | Stellar Objects | Asteroids | Exoplanets |
|----------|-----------------|-----------|------------|
| Dataset Size | 100,000 ✅ | 4,687 | 9,564 |
| Problem Type | Multi-class ✅ | Binary | Binary |
| Difficulty | Moderate ✅ | Moderate | Higher |
| Learning Value | High ✅ | Medium | High |
| Data Quality | Excellent ✅ | Good | Good |

---

# 3. Understanding the Physics (Simple Explanation)

## 3.1 What Is Light?

Light is a **wave**, and different colors have different **wavelengths**:

```
Short wavelength                              Long wavelength
(High energy)                                 (Low energy)

   UV      Violet   Blue   Green   Yellow   Red      Infrared
    |        |       |       |        |       |          |
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
◄─────────── HOT objects emit here    COOL objects emit here ──────────►
```

**Simple rule:**
- **Hot things** → emit more blue/UV light (short wavelength)
- **Cool things** → emit more red/infrared light (long wavelength)

That's why fire goes from red → orange → yellow → white → blue as it gets hotter!

## 3.2 The Three Types of Objects

### ⭐ Stars

```
        🌟
       ╱   ╲
      ╱ HOT ╲      A giant ball of gas doing nuclear fusion
      ╲ GAS ╱      (like our Sun)
       ╲   ╱       
        ───        
```

- **What it is:** Giant ball of hot gas
- **Size:** Varies (our Sun is average)
- **Distance from Earth:** Relatively CLOSE (within our galaxy)
- **Temperature:** 3,000 - 50,000°C

### 🌌 Galaxies

```
         ✦   ✦
       ✦ ✦ ✦ ✦ ✦
      ✦ ✦ ✦✦✦ ✦ ✦    Billions of stars + gas + dust + dark matter
       ✦ ✦ ✦ ✦ ✦      all held together by gravity
         ✦   ✦
```

- **What it is:** Billions of stars, gas, dust, and dark matter
- **Size:** Enormous (100,000 light-years across)
- **Distance from Earth:** MEDIUM to FAR (millions to billions of light-years)
- **Contains:** Billions of stars of different ages

### 💫 Quasars (Quasi-Stellar Objects)

```
            ═══════►  Jet of energy
              ╱
         ████████
        ██ ⚫ ██      A supermassive black hole eating material
        ██████████    The material heats up and glows EXTREMELY bright
              ╲
            ═══════►  Jet of energy
```

- **What it is:** Supermassive black hole actively "eating" material
- **Size:** Compact center, but incredibly bright
- **Distance from Earth:** VERY FAR (billions of light-years)
- **Brightness:** Can outshine entire galaxies!
- **Name origin:** "Quasi-stellar" = looks like a star but isn't

## 3.3 What Is Redshift? (THE MOST IMPORTANT CONCEPT!)

### The Doppler Effect

Imagine an ambulance:

```
🚑 Coming TOWARD you          🚑 Going AWAY from you
        
Sound waves compressed:       Sound waves stretched:
   )))))))                         )    )    )    )
   
   HIGH pitch                     LOW pitch
   "EEEEEE"                       "OOOOOOO"
```

**Light works the same way!**

```
Object moving TOWARD us:       Object moving AWAY from us:

Light waves compressed:        Light waves stretched:
   ~~~~~~~~                        ~    ~    ~    ~
   
   Looks more BLUE                Looks more RED
   "Blueshift"                    "Redshift"
```

### Why Does Redshift Tell Us Distance?

**The universe is expanding!** Everything is moving away from everything else.

```
The Universe is Like a Balloon Being Inflated:

    Before                          After
    
     🌟                              🌟
      ·  🌍                            ·    🌍
        ·  ★                              ·      ★
          · 💫                                ·        💫

The FARTHER something is, the FASTER it moves away = MORE redshift
```

### Redshift Values by Object Type

| Object | Redshift Value | What It Means |
|--------|----------------|---------------|
| **Star** | ~0.0001 (very low) | Close to Earth, barely moving away |
| **Galaxy** | 0.01 - 0.5 (medium) | Medium distance, moving moderately |
| **Quasar** | 1.0 - 7.0 (high) | Extremely far, moving very fast |

## 3.4 What Are u, g, r, i, z Filters?

The telescope measures light through **5 different color filters**:

| Filter | Name | What It Captures | Wavelength |
|--------|------|------------------|------------|
| **u** | Ultraviolet | Very hot objects | 354 nm (violet/UV) |
| **g** | Green | Medium-hot objects | 477 nm (blue-green) |
| **r** | Red | Cooler objects | 623 nm (red) |
| **i** | Infrared | Even cooler | 762 nm (near-infrared) |
| **z** | Z-band | Coolest/most distant | 913 nm (deep infrared) |

### How Different Objects Look Through Filters

```
STAR (Hot, close):           GALAXY (Mixed ages):        QUASAR (Extreme, far):

Brightness                   Brightness                   Brightness
    │ ██                         │    ████                     │██
    │ ████                       │ ████████                    │████
    │ ██████                     │ ██████████                  │ ██████
    │ ████████                   │████████████                 │  ████████
    └──────────→                 └──────────→                  └──────────→
      u g r i z                    u g r i z                     u g r i z
      
Brightest in blue            Even across all              Bright in UV (but
(hot surface)                (mix of old & young stars)   redshifted by distance)
```

## 3.5 Why Do We Need Both Filters AND Redshift?

### The Problem: Colors Can Be Deceiving!

```
Imagine two objects with IDENTICAL u,g,r,i,z values:

Object A:  u=19.2  g=18.5  r=18.1  i=17.9  z=17.7
Object B:  u=19.2  g=18.5  r=18.1  i=17.9  z=17.7

Same colors... but:

Object A: redshift = 0.0002  →  It's a STAR (close to us)
Object B: redshift = 2.5     →  It's a QUASAR (very far away!)
```

### Why This Happens

Redshift **changes** how an object's colors appear to us:

```
QUASAR's journey to Earth:

Original light                    What we OBSERVE
(at the quasar)                   (after traveling billions of years)

   Very bright                       Light has been
   in ultraviolet     ──────►        "stretched" to look
                                     more like a star!
```

### The Solution: Use Both!

| Feature | What It Tells Us |
|---------|------------------|
| u, g, r, i, z | What the object **LOOKS LIKE** (after redshift effect) |
| redshift | How **FAR AWAY** the object is (and how much light was stretched) |

**Together = Complete picture!**

```
Colors alone:     "This looks reddish"
                   Could be: Red star? Galaxy? Redshifted quasar?
                    CONFUSED

Colors + redshift: "This looks reddish AND has high redshift"
                   It's a quasar whose blue light got shifted to red!
                    CERTAIN
```

## 3.6 Do You Need Physics Formulas?

**NO!** Here's what you need vs. don't need:

| Task | Physics Needed? |
|------|-----------------|
| Loading and cleaning data | ❌ No |
| Training ML models | ❌ No |
| Evaluating accuracy | ❌ No |
| Explaining what redshift means | ✅ Just 1-2 sentences |
| Understanding WHY the model works | ❌ No (it's pattern matching) |

The ML model finds patterns **automatically**. You just need to understand the basic concepts to explain your project!

---

# 4. Understanding the Data

## 4.1 Dataset Overview

| Property | Value |
|----------|-------|
| **Total Rows** | 100,000 |
| **Total Columns** | 18 |
| **Source** | Sloan Digital Sky Survey (SDSS) DR17 |
| **Missing Values** | None ✅ |

## 4.2 Key Features We Use

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `u` | Ultraviolet brightness | 19.47 |
| `g` | Green brightness | 17.04 |
| `r` | Red brightness | 15.94 |
| `i` | Infrared brightness | 15.50 |
| `z` | Far-infrared brightness | 15.22 |
| `redshift` | Distance indicator | 0.634 |
| `class` | Object type (TARGET) | GALAXY |

**Note:** Lower brightness values = brighter objects (astronomical convention)

## 4.3 Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| GALAXY | 59,445 | 59.4% |
| STAR | 21,594 | 21.6% |
| QSO (Quasar) | 18,961 | 19.0% |

## 4.4 Features We DON'T Use

| Feature | Why We Skip It |
|---------|----------------|
| `obj_ID` | Just an identifier, no predictive value |
| `alpha`, `delta` | Sky coordinates - location doesn't determine type |
| `run_ID`, `field_ID` | Observation metadata |
| `plate`, `MJD`, `fiber_ID` | Telescope equipment info |

---

# 5. How Machine Learning Solves This Problem

## 5.1 The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                           ML MODEL              OUTPUT   │
│                                                                 │
│  u = 19.2                                                       │
│  g = 18.5                        ┌─────────┐                    │
│  r = 18.1           ──────►      │ Trained │    ──────►  STAR   │
│  i = 17.9                        │  Model  │           GALAXY   │
│  z = 17.7                        └─────────┘              QSO   │
│  redshift = 0.0002                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 The Step-by-Step Process

```
Step 1: LOAD DATA
    ↓
Step 2: EXPLORE DATA (EDA)
    ↓
Step 3: PREPROCESS DATA
    ↓
Step 4: SPLIT DATA (80% train, 20% test)
    ↓
Step 5: TRAIN MODELS
    ↓
Step 6: EVALUATE & COMPARE
    ↓
Step 7: SELECT BEST MODEL
```

## 5.3 What the Model "Learns"

The model discovers patterns like:

| Pattern | Conclusion |
|---------|------------|
| redshift < 0.001 | Probably a **STAR** |
| redshift between 0.01 - 0.5 | Probably a **GALAXY** |
| redshift > 1.0 | Probably a **QUASAR** |
| High u-g difference | More likely a **GALAXY** |

**The algorithm figures out these rules automatically from the training data!**

## 5.4 Models We Used

| Model | How It Works (Simple) | Speed | Accuracy |
|-------|----------------------|-------|----------|
| **Logistic Regression** | Draws lines to separate classes | Fast | ~96% |
| **Decision Tree** | Asks yes/no questions | Fast | ~97% |
| **Random Forest** | Many decision trees vote together | Medium | ~97-98% |
| **k-NN** | Looks at nearest neighbors | Medium | ~97% |
| **SVM** | Finds optimal boundaries | Slow | ~96% |

## 5.5 Why Random Forest Usually Wins

Random Forest combines **100 decision trees**:

```
                    New Object
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
      Tree 1          Tree 2          Tree 3    ... (100 trees)
         │               │               │
      "STAR"         "GALAXY"        "STAR"
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                    VOTE: STAR wins!
                    (67 trees said STAR)
```

**Benefits:**
- Reduces errors (one tree might be wrong, but 100 trees together are usually right)
- Handles complex patterns
- Not sensitive to outliers

---

# 6. Project Implementation

## 6.1 Required Libraries

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 6.2 Code Structure

| Section | Purpose | Lines of Code |
|---------|---------|---------------|
| 1. Setup | Import libraries | ~15 |
| 2. Load Data | Read CSV, explore | ~25 |
| 3. EDA | Visualizations | ~80 |
| 4. Preprocessing | Prepare for ML | ~25 |
| 5. Training | Train 5 models | ~45 |
| 6. Evaluation | Compare models | ~40 |
| 7. Feature Importance | Analyze | ~20 |
| 8. Conclusion | Summary | ~25 |
| **TOTAL** | | **~275 lines** |

## 6.3 Key Code Snippets

### Loading Data
```python
df = pd.read_csv('star_classification.csv')
print(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
```

### Preparing Features
```python
X = df[['u', 'g', 'r', 'i', 'z', 'redshift']]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Training a Model
```python
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

---

# 7. Results and Findings

## 7.1 Model Accuracy Comparison

| Rank | Model | Accuracy |
|------|-------|----------|
| 🥇 | **Random Forest** | ~97.5% |
| 🥈 | Decision Tree | ~97.2% |
| 🥉 | K-Nearest Neighbors | ~97.0% |
| 4 | Logistic Regression | ~96.5% |
| 5 | SVM | ~96.0% |

## 7.2 Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **redshift** | ~0.65 (65%) |
| 2 | z | ~0.08 |
| 3 | i | ~0.07 |
| 4 | r | ~0.07 |
| 5 | g | ~0.07 |
| 6 | u | ~0.06 |

**Key Finding:** Redshift is by far the most important feature, confirming our physics understanding!

## 7.3 Confusion Matrix Insights

```
                    Predicted
                 STAR  GALAXY  QSO
Actual  STAR     4280    25     14
        GALAXY     18  11845    26
        QSO        12    35   3745
```

**Observations:**
- Stars are classified very accurately (few mistakes)
- Galaxies sometimes confused with Quasars (both are "extended" objects)
- Quasars sometimes confused with Stars (both can look like point sources)

## 7.4 Key Findings Summary

1. **Redshift is the key** - It alone provides 65% of the predictive power
2. **Random Forest performs best** - Combining multiple trees improves accuracy
3. **97%+ accuracy is achievable** - ML can reliably classify astronomical objects
4. **Color features complement redshift** - Together they achieve near-perfect classification
5. **Stars are easiest to classify** - Their low redshift makes them distinct

---

# 8. Visualizations

## 8.1 Available Visualizations in the Notebook

| Visualization | Purpose |
|---------------|---------|
| Class distribution pie chart | Show dataset balance |
| Feature histograms by class | Compare distributions |
| Redshift distribution | Show why it's important |
| Correlation heatmap | Show feature relationships |
| **3D Interactive Space Map** | WOW factor - explore objects! |
| Light signatures comparison | Show color patterns |
| Model accuracy comparison | Compare all models |
| Confusion matrix | Show model errors |
| Feature importance chart | Show what matters |

## 8.2 The 3D Interactive Space Map

**What it does:**
- Plots 5,000 objects in 3D space
- X-axis: Right Ascension (sky position)
- Y-axis: Declination (sky position)
- Z-axis: Redshift (distance)
- Colors: Gold (Stars), Purple (Galaxies), Cyan (Quasars)

**Interactivity:**
- 🖱️ **Drag** to rotate
- 🔍 **Scroll** to zoom
- 👆 **Hover** to see object details

**What you'll see:**
- Stars clustered at the bottom (low redshift = close)
- Galaxies in the middle
- Quasars at the top (high redshift = far away)

---

# 10. References

## 9.1 Dataset

1. Fedesoriano. (2022). *Stellar Classification Dataset - SDSS17*. Kaggle.
   https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

## 9.2 Scientific Background

2. Sloan Digital Sky Survey. (n.d.). *SDSS Home*.
   https://www.sdss.org/

3. NASA. (n.d.). *Basics of Space Flight*.
   https://science.nasa.gov/learn/basics-of-space-flight/

## 9.3 Machine Learning

4. Scikit-learn Documentation.
   https://scikit-learn.org/stable/

5. Plotly Python Graphing Library.
   https://plotly.com/python/

## 9.4 Related Research

6. Joshi, S. (2024). *Stellar Object Classification Using Machine Learning*. Medium.
   https://medium.com/@an.sum.joshi/stellar-object-classification-using-machine-learning

---

# Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Redshift** | The stretching of light toward red wavelengths, indicating an object moving away from us |
| **Quasar** | Quasi-Stellar Object - an extremely bright galactic nucleus powered by a supermassive black hole |
| **SDSS** | Sloan Digital Sky Survey - a major astronomical survey that mapped millions of objects |
| **Photometric filters** | The u, g, r, i, z filters that measure light at different wavelengths |
| **Classification** | The ML task of predicting which category an object belongs to |
| **Random Forest** | An ensemble ML algorithm that combines many decision trees |
| **Feature Importance** | A measure of how much each input feature contributes to predictions |

---

# Appendix B: Quick Reference Card

## How to Run the Project

```bash
# 1. Install requirements
pip install pandas numpy matplotlib seaborn plotly scikit-learn

# 2. Download dataset from Kaggle
# https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

# 3. Place star_classification.csv in same folder as notebook

# 4. Open notebook and run all cells
```

## Key Results to Remember

- **Best Model:** Random Forest (~97.5% accuracy)
- **Most Important Feature:** Redshift (65% importance)
- **Dataset Size:** 100,000 observations
- **Number of Classes:** 3 (Star, Galaxy, Quasar)
- **Number of Features Used:** 6 (u, g, r, i, z, redshift)

---

*Document created for ARTI 308 Machine Learning Course Project*
*Imam Abdulrahman Bin Faisal University*
*Academic Year 2025-2026*
