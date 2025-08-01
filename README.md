StockSage - AI Stock Prediction

## Description

StockSage is a web application built with Flask that predicts future stock prices using a Long Short-Term Memory (LSTM) neural network. It fetches historical stock data from Alpha Vantage, trains an LSTM model on the fly, predicts the next day's mid-price, and displays the prediction alongside a chart comparing predicted vs. actual past prices on a modern, interface.

## Features

* **Stock Price Prediction:** Enter a valid stock ticker symbol (e.g., NVDA, AAPL) to get a prediction for the next trading day's mid-price (average of High and Low).
* **LSTM Model:** Utilizes a Keras-based LSTM model trained on historical Open, High, Low, and Close prices.
* **Data Visualization:** Displays a line chart comparing the model's predictions on the test set against the actual historical mid-prices using Chart.js.
* **Dynamic Data Fetching:** Retrieves up-to-date daily stock data from the Alpha Vantage API.
* **API Key Rotation:** Cycles through multiple Alpha Vantage API keys provided in an environment file to mitigate rate limiting.
* **Modern UI:** Clean, responsive, dark-themed user interface built with custom CSS.

## Tech Stack

* **Backend:**
    * Python 3.x
    * Flask
    * Flask-RESTful
    * Flask-Cors
* **Machine Learning:**
    * TensorFlow / Keras
    * Pandas
    * NumPy
    * Scikit-learn
* **Data Source:**
    * Alpha Vantage API
* **Frontend:**
    * HTML5
    * CSS3
    * JavaScript
    * jQuery
    * Chart.js
* **Environment:**
    * `python-dotenv`

## Setup & Installation

1.  **Prerequisites:**
    * Python 3.7+ installed.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/hk151109/StockSage
    cd StockSage
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```txt
    flask
    flask-restful
    flask-cors
    python-dotenv
    tensorflow
    numpy
    pandas
    scikit-learn
    ```

5.  **Set Up Environment Variables:**
    * Create a file named `.env` in the root directory of the project.
    * Add your Alpha Vantage API keys as a comma-separated string:
        ```dotenv
        ALPHAVANTAGE_API_KEYS="YOUR_KEY_1,YOUR_KEY_2,YOUR_KEY_3"
        ```
    * Replace `YOUR_KEY_1`, etc., with your actual keys. You can add one or more keys. The application will rotate through them. Get free keys from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

## Running the Application

1.  **Ensure your virtual environment is activated.**
2.  **Run the Flask development server:**
    ```bash
    python app.py
    ```
3.  **Open your web browser** and navigate to: `http://127.0.0.1:5000`.

## Project Structure


## API Key Rotation

The application loads Alpha Vantage API keys from the `ALPHAVANTAGE_API_KEYS` variable in the `.env` file. For each prediction request that requires fetching new data from Alpha Vantage, the application uses the next key in the list, wrapping around to the beginning when it reaches the end. This helps distribute API calls across multiple keys if you are making frequent requests.

## Notes

* The LSTM model is trained *every time* a prediction is requested for a ticker whose data isn't cached locally as a CSV. This can be time-consuming, especially the first time for a given ticker.
* The model architecture and hyperparameters in `lstm.py` are basic; performance can likely be improved with further tuning.
* The temporary `stock_market_data-TICKER.csv` file created during data fetching is deleted after the prediction is complete.


## Potential Improvements

* Fixed Window Size: The sequence length (window = 5) is hardcoded. Making this configurable could help optimize performance.
* Limited Hyperparameter Tuning: The model uses fixed batch size (512) and epochs (20) without early stopping or learning rate scheduling.
* Simple Feature Set: While using OHLC + Mid Price is standard, adding technical indicators (RSI, MACD, etc.) or external factors could improve predictions.
* No Backtesting: The code lacks a comprehensive backtesting framework to evaluate trading performance based on predictions. Cross-Validation: Implement time-series cross-validation instead of a simple train/test split.

