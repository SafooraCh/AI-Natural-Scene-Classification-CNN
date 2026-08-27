
````
# 🌍 AI Natural Scene Classification Using CNN

An AI-based image classification system that uses a **Convolutional Neural Network (CNN)** to automatically classify images into six different scene categories.

## 📌 Project Overview

This project uses the **Intel Image Classification Dataset** to train a custom CNN model from scratch.

The model takes an image as input and predicts which type of scene it belongs to.

### 🏷️ Supported Classes

- 🏢 Buildings
- 🌲 Forest
- ❄️ Glacier
- ⛰️ Mountain
- 🌊 Sea
- 🛣️ Street

---

## 🎯 Objective

The main objective of this project is to develop a Deep Learning model that can automatically recognize and classify different natural and urban scenes from images.

The project demonstrates the practical implementation of:

- Image preprocessing
- Data augmentation
- Convolutional Neural Networks
- Model training
- Model validation
- Model evaluation
- Image prediction
- Confusion matrix
- Classification report

---

## 📊 Dataset

### Intel Image Classification Dataset

The project uses the Intel Image Classification dataset available on Kaggle.

**Dataset:** Intel Image Classification

**Source:** Kaggle

The dataset contains approximately 25,000 images belonging to six scene categories.

---

## 🧠 Model

A **custom CNN model** is developed from scratch.

### CNN Architecture

```text
Input Image
150 × 150 × 3
       ↓
Conv2D - 32 Filters
       ↓
ReLU
       ↓
MaxPooling
       ↓
Conv2D - 64 Filters
       ↓
ReLU
       ↓
MaxPooling
       ↓
Conv2D - 128 Filters
       ↓
ReLU
       ↓
MaxPooling
       ↓
Flatten
       ↓
Dense - 128
       ↓
ReLU
       ↓
Dropout
       ↓
Dense - 6
       ↓
Softmax
````

### Important

This project uses **CNN only**.

It does not use:

* ResNet
* VGG
* MobileNet
* EfficientNet
* Transfer Learning
* Grad-CAM

---

## ⚙️ Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming          |
| TensorFlow   | Deep Learning        |
| Keras        | CNN Development      |
| NumPy        | Numerical Operations |
| Matplotlib   | Visualization        |
| Seaborn      | Confusion Matrix     |
| Scikit-learn | Model Evaluation     |
| Pillow       | Image Processing     |
| Streamlit    | Web Application      |

---

## 📁 Project Structure

```text
AI-Natural-Scene-Classification-CNN/
│
├── app.py
│
├── intel_scene_cnn.keras
│
├── intel_scene_cnn_info.pkl
│
├── requirements.txt
│
├── README.md
│
└── notebook/
    └── Intel_Image_Classification_CNN_Kaggle.ipynb
```

---

## 🚀 Features

### 1. Image Upload

The user can upload an image through the Streamlit application.

### 2. Scene Classification

The CNN predicts one of the six scene categories.

### 3. Confidence Score

The application displays the confidence score of the prediction.

Example:

```text
Prediction: Mountain

Confidence: 94.32%
```

### 4. Class Probabilities

The application also displays the probability for each scene class.

Example:

```text
Buildings  → 1.20%
Forest     → 2.45%
Glacier    → 0.81%
Mountain   → 94.32%
Sea        → 0.72%
Street     → 0.50%
```

### 5. Model Evaluation

The project includes:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Test accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## 💾 Saved Model Files

### CNN Model

```text
intel_scene_cnn.keras
```

This file contains the trained CNN model.

### PKL File

```text
intel_scene_cnn_info.pkl
```

This file stores supporting model information such as:

* Class names
* Class indices
* Image size
* Number of classes
* Model type
* Dataset information
* Image normalization information

---

## 🖥️ Run the Application

### Step 1 — Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Natural-Scene-Classification-CNN.git
```

### Step 2 — Open Project

```bash
cd AI-Natural-Scene-Classification-CNN
```

### Step 3 — Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4 — Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

```text
streamlit
tensorflow
numpy
pillow
matplotlib
seaborn
scikit-learn
```

---

## 🔬 Model Training

The CNN is trained using the Intel Image Classification dataset.

### Training Process

```text
Dataset
   ↓
Image Loading
   ↓
Image Resizing
   ↓
Normalization
   ↓
Data Augmentation
   ↓
Train / Validation Split
   ↓
CNN Training
   ↓
Model Evaluation
   ↓
Save Model
```

---

## 📈 Evaluation

The model is evaluated on unseen test images.

The notebook generates:

### Accuracy Graph

Shows training and validation accuracy across epochs.

### Loss Graph

Shows training and validation loss across epochs.

### Confusion Matrix

Shows correct and incorrect predictions for all six classes.

### Classification Report

Provides:

```text
Precision
Recall
F1-Score
Support
```

---

## 🧪 Prediction Example

The application accepts an image:

```text
Input Image
     ↓
Preprocessing
     ↓
CNN
     ↓
Prediction
```

Example output:

```text
Predicted Class: Forest

Confidence: 91.52%
```

---

## 🔮 Future Improvements

Possible future improvements include:

* Real-time camera classification
* Mobile application
* Cloud deployment
* More scene categories
* Larger training dataset
* Improved CNN architecture
* Real-time prediction

---

## 👩‍💻 Author

**Safoora**

BS Artificial Intelligence
Sir Syed CASE Institute of Technology

---

## 📜 License

This project is developed for **educational and academic purposes**.

```

**Repository name:** `AI-Natural-Scene-Classification-CNN`

One important recommendation: **don't upload a huge dataset folder to GitHub**. Keep the Kaggle dataset separate and upload your notebook, `app.py`, requirements, README, and model files (if their size fits GitHub's limits).
```
