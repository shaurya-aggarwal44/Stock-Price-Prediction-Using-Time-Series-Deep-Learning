# 📈 Stock Price Prediction Using Time Series & Deep Learning

## Overview
This project implements and compares multiple **time series forecasting models** to predict the **next-day closing price of a stock**. Both **classical statistical approaches** and **deep learning models** are evaluated to study the impact of model complexity on prediction accuracy.

The project highlights how **simple baseline models can sometimes outperform complex neural networks**, while also demonstrating the effectiveness of **LSTM-based architectures** for financial time series forecasting.

<p align="center">
  <img src="/Images/spy_plot.png" alt="Stock Price Chart">
</p>

---

## Key Objectives
- Build an end-to-end **time series forecasting pipeline**
- Compare **statistical models vs deep learning models**
- Analyze model performance using **Mean Absolute Error (MAE)**
- Visualize predictions, trends, and errors
- Identify the most effective model for stock price prediction

---

## Dataset
- **Source:** Yahoo Finance (`yfinance` API)
- **Asset:** SPY ETF
- **Time Period:** Jan 1993 – Sep 2020
- **Frequency:** Daily
- **Type:** Univariate time series (~7,000 data points)

The dataset is split into **training, validation, and testing sets**.

<p align="center">
  <img src="/Images/SPY_train_valid_test_plot.png" alt="Data Split">
</p>

---

## Models Implemented

### 1. Naive Forecast
- Uses previous day’s closing price as prediction
- Strong baseline due to low daily volatility

<p align="center">
  <img src="/Images/naive_forecast_plot.png">
</p>

---

### 2. Moving Average Models
- 5-day and 20-day Simple Moving Averages
- Used for trend analysis and smoothing

<p align="center">
  <img src="/Images/20_day_ma_plot.png">
</p>

---

### 3. ARIMA Model
- AutoRegressive Integrated Moving Average
- Configuration: **ARIMA(1,1,1)**
- Includes stationarity and trend-seasonality analysis

<p align="center">
  <img src="/Images/arima_predictions.png">
</p>

---

### 4. Linear & Dense Neural Networks
- Linear regression using a single dense neuron
- Multi-layer dense neural network using Keras/TensorFlow

<p align="center">
  <img src="/Images/dense_forecast.png">
</p>

---

### 5. Recurrent Neural Network (RNN)
- Sequence-to-vector and sequence-to-sequence models
- Captures temporal dependencies in time series data

<p align="center">
  <img src="/Images/rnn_forecast.png">
</p>

---

### 6. LSTM Model
- Long Short-Term Memory network
- Best performing model
- 30-day rolling window achieved lowest MAE

<p align="center">
  <img src="/Images/lstm_30day_window.png">
</p>

---

### 7. CNN-Based Models
- CNN preprocessing combined with RNN
- Full CNN using a WaveNet-style architecture

<p align="center">
  <img src="/Images/full_cnn_wavenet.png">
</p>

---

## Model Performance Comparison
Mean Absolute Error (MAE) across all models:

<p align="center">
  <img src="/Images/model_results.png">
</p>

---

## Tech Stack
- Python
- NumPy, Pandas
- Matplotlib
- Statsmodels (ARIMA)
- TensorFlow / Keras
- Google Colab

---

## Key Takeaways
- Simple baseline models can outperform complex architectures
- LSTM models are highly effective for financial time series
- Proper window size selection significantly impacts performance

---

## How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
