import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    
    le = LabelEncoder()
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
        
    df = pd.get_dummies(df, columns=['gender', 'MultipleLines', 'InternetService', 
                                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                     'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                     'Contract', 'PaymentMethod'])
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    joblib.dump(sc, 'scaler.pkl')
    
    return X_train, X_test, y_train, y_test

def build_model(input_dim):
    model = Sequential([
        Dense(units=16, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(units=8, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('Telco-Customer-Churn.csv')
    
    model = build_model(X_train.shape[1])
    
    print("Starting Training...")
    model.fit(X_train, y_train, 
              epochs=50, 
              batch_size=32, 
              validation_split=0.2, 
              verbose=1)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    model.save('churn_model.h5')
    print("Model and Scaler saved successfully.")