# CT Scan COVID-19 Classification Using Convolutional Neural Networks

## 1. Project Overview

This project implements and evaluates multiple Convolutional Neural Network (CNN) architectures to classify chest CT scan images into two categories:

- COVID-19 Positive
- Non-COVID

The primary objective is to explore deep learning techniques for medical image classification, systematically compare different model architectures, and evaluate the impact of hyperparameters such as dropout rates and activation functions.

This is a personal academic project developed to strengthen practical understanding of CNN design, regularization, and experimental evaluation.

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

To address overfitting, dropout layers were introduced and evaluated with varying rates using:

runNNModel_DropOut(D)

The goal was to determine the optimal regularization level for better generalization on unseen data.

---

### 4.3 Activation Function Experiments

Different activation functions were tested using:

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

- Deeper architectures improve feature extraction but increase overfitting risk.
- Dropout significantly improves generalization.
- Activation functions influence training stability and convergence speed.
- Well-regularized simpler models can perform competitively.

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

---

## 10. Conclusion

This project demonstrates practical implementation of CNN-based medical image classification, including:

- Model architecture experimentation
- Hyperparameter tuning
- Regularization strategies
- Comparative evaluation

It provides hands-on experience in designing, training, and analyzing deep learning models in a healthcare context.
