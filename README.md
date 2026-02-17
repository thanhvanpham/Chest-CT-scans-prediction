# 🧠 CT Scan COVID Classifier  
A Deep Learning Exploration in Medical Image Classification  

## 1. Project Overview

This project develops a Convolutional Neural Network (CNN) model to classify chest CT scan images into two categories:

- 🦠 COVID Positive  
- 😌 Non-COVID  

The goal of this project was to build, experiment with, and evaluate multiple CNN architectures to better understand deep learning model behavior, regularization techniques, and performance trade-offs in medical image classification.

This is a personal learning project focused on strengthening practical knowledge in computer vision and neural network experimentation.

---

## 2. Dataset

The dataset consists of CT scan images organized into two folders:

- CT_COVID/      → COVID-19 positive cases
- CT_NonCOVID/   → Non-COVID cases

### Preprocessing Steps

- Images resized to 128 x 128 pixels
- Pixel values normalized
- Binary label encoding
- Train/test split using scikit-learn

All preprocessing ensures consistent input dimensions and stable model training.

---

## 3. Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- scikit-image
- Jupyter Notebook

---

## 4. Methodology

### 4.1 CNN Architecture Design

Multiple CNN models were implemented and compared. Core components include:

- Conv2D layers
- MaxPooling layers
- Flatten layer
- Dense (fully connected) layers
- Sigmoid output layer (binary classification)

Models varied in depth and complexity to evaluate how architecture changes impact performance.

General structure:

Conv2D
MaxPooling
Conv2D
MaxPooling
Flatten
Dropout
Dense
Output (Sigmoid)

---

### 4.2 Dropout Experiments

To address overfitting, I created a function:

runNNModel_DropOut(D)

And tested different dropout rates. Without dropout, the model gets too attached to the training data.

---

### 4.3 Activation Function Experiments

I also experimented with:

runNNModel_ActivationFunctions(A)

Primary focus included:
- ReLU
- Sigmoid
- Other potential alternatives

This allowed comparison of convergence behavior and model stability.

---

### 4.4 Final Model Selection

Final models were selected based on:

- Validation accuracy
- Stability of training and validation loss curves
- Reduced overfitting
- Balanced performance between train and test sets

Two final configurations were evaluated in detail.

---

## 5. Evaluation Metrics

Model performance was assessed using:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Loss curves
- Runtime comparison

Performance curves were plotted to analyze convergence behavior and overfitting patterns.

---

## 6. Key Findings

- CNNs are powerful even with simple architectures.
- Dropout actually matters.
- Activation functions affect performance more than expected.
- Overfitting is sneaky.
- Training time increases when you get ambitious.
- Debugging TensorFlow at 2AM is an unforgettable experience :)

---

## 7. Limitations

- Dataset size may limit generalization.
- No cross-validation implemented.
- Evaluation primarily focused on accuracy.
- Precision, recall, F1-score, and confusion matrix not included.
- Transfer learning was not fully implemented.

---

## 8. Future Improvements

- Implement transfer learning (e.g., MobileNetV2, ResNet)
- Add data augmentation
- Apply early stopping
- Include additional evaluation metrics (precision, recall, F1-score, ROC curve)
- Perform cross-validation
- Deploy as a simple demonstration web application

---

## 9. Disclaimer

This project is developed for educational and research purposes only.

It is not intended for clinical diagnosis and should not be used for medical decision-making.

However, it should provide hands-on experience in designing, training, and analyzing deep learning models in a healthcare context.
