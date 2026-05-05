# 📊 Customer Churn Prediction System

A machine learning application that uses an **Artificial Neural Network (ANN)** to predict the likelihood of a customer leaving a service based on their usage patterns and demographics.

## 🚀 Live Demo
You can access the live application here: 
**[Insert Your Streamlit URL Here]**

## 📝 Project Overview
Customer churn is a critical metric for service-based businesses. This project implements a Deep Learning approach to identify high-risk customers, allowing businesses to take proactive retention measures.

The model is trained on the **Telco Customer Churn** dataset and achieves high accuracy by capturing non-linear relationships between features like tenure, contract type, and monthly charges.

## 🛠️ Tech Stack
*   **Deep Learning:** Artificial Neural Network (ANN) via TensorFlow/Keras
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Web Interface:** Streamlit
*   **Serialization:** Joblib (for saving the StandardScaler)

## 🏗️ Model Architecture
The ANN consists of:
1.  **Input Layer:** 40+ features (after One-Hot Encoding).
2.  **Hidden Layer 1:** 16 neurons with **ReLU** activation.
3.  **Dropout Layer:** 20% dropout to prevent overfitting.
4.  **Hidden Layer 2:** 8 neurons with **ReLU** activation.
5.  **Output Layer:** 1 neuron with **Sigmoid** activation for binary classification.

## DEMO LINK 
https://customer-churn-prediction-cb2zjdz7cxaimqxjgttlbe.streamlit.app/
