# 📈 MLStock – Stock Price Prediction Model (Based on Taiwan Stocks)

This project aims to establish a prediction system using a deep learning model to predict whether the stocks will rise or fall by more than a specific target percentage the next day based on technical indicators, prices, trading volumes and other data.

---

## 🎯 Project Goal

- Predict "whether a single stock will rise or fall by more than a specific target percentage the next day"
- In this project, specific target percentage is usually 2%.
- If the prediction target is "rise", the model needs to predict whether the highest price of the next day is > opening price * [specific target percentage],
- If the prediction target is "fall", the model predicts whether the lowest price is < opening price * [1-specific target percentage]
- The prediction method is binary classification (yes/no), and the model needs to be trained separately (one model for rise and fall)

---

## 🧠 Used Model

- `Fully Connected Network`
- `CNN + LSTM`
- `Temporal Convolutional Network (TCN)`

---

## 🧪 Features and Data Processing

- Technical indicators that users can choose to use (such as RSI, MACD, MA, trading volume, etc.)
- Features include price, trading volume, technical indicators and other data
- Feature standardization (Z-score)
- Supports training in units of "industry" or "stock"

---

## 🏗️ Project architecture

```plaintext
mlstock/
├── data_processing/ # Feature engineering and data processing module
├── models/ # Model definitions: FCN / CNN-LSTM / TCN
├── main.ipynb # Main Notebook (model training and testing process)
└── README.md
