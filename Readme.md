# 🧠 Predicting Diabetes Progression Using ML & Deep Learning

## ✅ Objective

The goal of this project is to **predict diabetes progression** using patient medical data by building and evaluating multiple machine learning and deep learning models.

---

## 📊 Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, containing health measurements like:

- Glucose
- Blood Pressure
- BMI
- Insulin levels
- Age, etc.

**Target Column**: `Outcome`
(1 = Diabetic, 0 = Non-diabetic)

---

## 🔍 Step 1: Data Understanding & Cleaning

### ✅ Tasks Performed:

- Loaded dataset using `pandas`
- Replaced invalid 0s with NaN in:
  - Glucose, BloodPressure, SkinThickness, Insulin, BMI
- Handled missing values using **median imputation**
- Performed data inspection using `.info()`, `.describe()`, and `.isnull()`
- Visualized correlation using seaborn heatmap and pairplots

---

## 🧪 Step 2: Feature Engineering

### ➕ New Features Added:

- `Obese`: Binary feature (BMI > 30)
- `GIR_log`: Log of Glucose-to-Insulin Ratio
- `Glucose_Age_Interaction`: Glucose × Age interaction

### ✅ Feature Selection:

- Used `mutual_info_classif()` to identify top features
- Selected 8 most important features:
  - Glucose_Age_Interaction, Glucose, BMI, Age, Pregnancies, Insulin, SkinThickness, Obese

---

## 🤖 Step 3: Classical Machine Learning Models

### 📌 Models Trained:

- **Decision Tree**
- **Random Forest**
- **XGBoost**

### 🔍 Evaluation Metrics:

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**
- **ROC-AUC Score**
- **Feature Importance** (for XGBoost)

📈 **ROC Curve** was plotted and saved using `matplotlib`.

---

## 🧠 Step 4: Deep Learning Model (TensorFlow/Keras)

### 🏗️ Model Architecture:

- Dense(64, relu)
- Dense(32, relu)
- Dense(1, sigmoid)

### ✅ Training Strategy:

- Loss: `binary_crossentropy`
- Optimizer: `Adam`
- Metrics: `Accuracy`
- **EarlyStopping** & **ModelCheckpoint** used

### 📊 Visualizations:

- Training vs. Validation Loss
- Training vs. Validation Accuracy

### 📌 Deep Learning Evaluation:

- Confusion Matrix
- Classification Report
- ROC-AUC Score

---

## 🚀 Conclusion

This project demonstrates how combining **feature engineering**, **ML models**, and **deep learning** can lead to accurate disease prediction. It can be extended by:

- Hyperparameter tuning
- Cross-validation
- Model ensembling
- Deploying the model as a web app

---

## 📦 Libraries Used

- pandas, numpy, seaborn, matplotlib
- scikit-learn
- xgboost
- tensorflow / keras

---

## 💼 Author

**Muhammad Nadeem**
BS Bioinformatics | Data Science Enthusiast
📧 [nadeem62354@gmail.com]
🔗 [https://www.linkedin.com/in/m-nadeem-655664284/]
🔗 [https://github.com/Nadeem786087]

---
