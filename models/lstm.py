import numpy as np
import pandas as pd
from dotenv import load_dotenv
import itertools
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
import time
import datetime as dt
import urllib.request
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

load_dotenv() #load the env variables

class LSTM_Model:

    # --- Add these lines inside the LSTM_Model class ---

    # Class attributes to store keys and the current index for rotation
    api_keys = []
    current_key_index = 0

    @classmethod
    def _load_api_keys_from_env(cls):
        """Loads API keys from the .env file if not already loaded."""
        if not cls.api_keys:  # Load only once
            load_dotenv() # Ensure .env is loaded
            keys_str = os.getenv("ALPHAVANTAGE_API_KEYS")
            if keys_str:
                # Split by comma and remove any leading/trailing whitespace from keys
                cls.api_keys = [key.strip() for key in keys_str.split(',') if key.strip()]
                print(f"Loaded {len(cls.api_keys)} Alpha Vantage API keys.")
            else:
                print("Warning: 'ALPHAVANTAGE_API_KEYS' not found or empty in .env file.")
            # Reset index when loading keys
            cls.current_key_index = 0

    @classmethod
    def _get_next_api_key(cls):
        """Gets the next API key from the list, rotating back to the start."""
        cls._load_api_keys_from_env() # Ensure keys are loaded

        if not cls.api_keys:
            # Handle the case where no keys are available
            # Option 1: Raise an error
            raise ValueError("No Alpha Vantage API keys configured or loaded.")
            # Option 2: Return None or a default/demo key (less safe)
            # print("Error: No API keys available.")
            # return None # Or return 'demo' - but the API might reject 'demo' often

        # Get the key at the current index
        key = cls.api_keys[cls.current_key_index]

        # Move to the next index, wrapping around using modulo
        cls.current_key_index = (cls.current_key_index + 1) % len(cls.api_keys)

        # Optional: Print which key index is being used (for debugging)
        print(f"Using API key index: {(cls.current_key_index - 1 + len(cls.api_keys)) % len(cls.api_keys)}")

        return key



    @classmethod
    def LSTM_Pred(cls, tick):

        data_source = 'alphavantage'

        if data_source == 'alphavantage':
            # ====================== Loading Data from Alpha Vantage ==================================

            # Get the next API key from the environment variable list
            try:
                api_key = cls._get_next_api_key() # <<< ADD THIS LINE
            except ValueError as e:
                print(f"API Key Error: {e}")
                # Cannot proceed without an API key
                return None, None, None, None

            ticker = tick

            url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" % (
                ticker, api_key)

            file_to_save = 'stock_market_data-%s.csv' % ticker

            if not os.path.exists(file_to_save):
                try:
                    with urllib.request.urlopen(url_string) as url:
                        data = json.loads(url.read().decode())
                        # extract stock market data
                        if 'Time Series (Daily)' not in data:
                             print(f"Error fetching data from Alpha Vantage for {ticker}.")
                             print(f"Response: {data}")
                             # Handle error appropriately, maybe raise an exception or return None
                             return None, None, None, None # Or some other error indication

                        data = data['Time Series (Daily)']
                        df = pd.DataFrame(
                            columns=['Date', 'Low', 'High', 'Close', 'Open'])
                        # Data loading improvement: collect rows in list first
                        rows_list = []
                        for k, v in data.items():
                            date = dt.datetime.strptime(k, '%Y-%m-%d')
                            # Check if all expected keys are present
                            if all(key in v for key in ['3. low', '2. high', '4. close', '1. open']):
                                data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                                            float(v['4. close']), float(v['1. open'])]
                                rows_list.append(data_row)
                            else:
                                print(f"Warning: Missing data for date {k}. Skipping this entry.")

                        # Create DataFrame from list of lists
                        temp_df = pd.DataFrame(rows_list, columns=['Date', 'Low', 'High', 'Close', 'Open'])
                        # Append to the main df (though creating directly is better)
                        # Sticking to original logic structure as much as possible:
                        df = pd.concat([df, temp_df], ignore_index=True)

                    print('Data saved to : %s' % file_to_save)
                    # Ensure Date column is correct type before saving
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date') # Sort before saving
                    df.to_csv(file_to_save, index=False) # Save without index

                except urllib.error.URLError as e:
                    print(f"Error opening URL: {e}")
                    return None, None, None, None # Or some other error indication
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON response: {e}")
                    return None, None, None, None # Or some other error indication
                except Exception as e:
                    print(f"An unexpected error occurred during data fetching: {e}")
                    return None, None, None, None # Or some other error indication


            # If the data is already there, just load it from the CSV
            else:
                print('File already exists. Loading data from CSV')
                df = pd.read_csv(file_to_save, parse_dates=['Date']) # Parse dates when loading

        # Ensure file_name corresponds to the saved file
        file_name = 'stock_market_data-%s.csv' % ticker # Use the same format as above
        if not os.path.exists(file_name):
             print(f"CSV file {file_name} not found. Cannot proceed.")
             return None, None, None, None

        # Read again to ensure consistency, or rely on df from above if just created
        df = pd.read_csv(file_name, parse_dates=['Date'])

        df = df[['Date', 'Open', 'Close', 'Low', 'High']]

        df = df.sort_values('Date') # Sort by date

        # Check for NaNs after loading and sorting
        if df.isnull().values.any():
            print("Warning: NaN values found in data. Attempting to fill.")
            # Simple forward fill, consider more sophisticated methods if needed
            df.fillna(method='ffill', inplace=True)
            # Check if NaNs still exist (e.g., at the beginning)
            if df.isnull().values.any():
                print("Error: NaNs still present after forward fill. Cannot proceed.")
                return None, None, None, None

        high_prices = df.loc[:, 'High']
        low_prices = df.loc[:, 'Low']
        df["Mid Prices"] = (high_prices + low_prices) / 2.0

        # Keep date for potential future use or indexing, but remove for scaling
        dates = df['Date'] # Store dates if needed later
        df.drop("Date", axis=1, inplace=True)

        df1 = df.copy() # Use copy to avoid modifying df1 when df is scaled
        df_values = df.values # Use values for scaling

        # Check if data is empty before scaling
        if df_values.shape[0] < 10: # Need enough data for sequence + train/test split
             print(f"Error: Insufficient data points ({df_values.shape[0]}) for processing.")
             return None, None, None, None

        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_values)

        # --- Inner Helper Functions ---
        def build_model(input_shape_tuple): # Pass input shape explicitly
            model = Sequential()

            model.add(LSTM(
                input_shape=input_shape_tuple, # Use the provided tuple
                units=50,
                return_sequences=True))
            model.add(Dropout(0.2))

            model.add(LSTM(
                100,
                return_sequences=False))
            model.add(Dropout(0.2))

            model.add(Dense(units=1))
            model.add(Activation('linear'))

            start = time.time()
            model.compile(loss='mse', optimizer='rmsprop') # Keep rmsprop as per original
            print('Compilation time : ', time.time() - start)
            return model

        def load_data(stock_scaled, seq_len):
            amount_of_features = stock_scaled.shape[1] # Get number of features dynamically
            data = stock_scaled
            sequence_length = seq_len + 1
            result = []
            # Ensure index range is valid
            if len(data) <= sequence_length:
                print("Error: Data length is too short for the sequence length.")
                return None # Indicate error

            for index in range(len(data) - sequence_length + 1): # Corrected range
                result.append(data[index: index + sequence_length])

            result = np.array(result)
            if result.shape[0] == 0:
                 print("Error: No sequences generated. Check data length and sequence length.")
                 return None # Indicate error

            row = round(0.75 * result.shape[0])
            if row == 0 or row == result.shape[0]: # Ensure train/test split is valid
                print("Warning: Train/Test split resulted in empty set. Adjusting split or check data.")
                # Handle edge case: maybe force at least one sample in test?
                row = max(1, int(result.shape[0] * 0.75)) if result.shape[0] > 1 else 0


            train = result[:int(row), :]
            test = result[int(row):, :] # Separate test set first

            if train.shape[0] == 0 or test.shape[0] == 0:
                 print(f"Error: Train ({train.shape[0]}) or Test ({test.shape[0]}) set is empty.")
                 return None # Indicate error


            x_train = train[:, :-1]
            y_train = train[:, -1][:, -1] # Target is the last feature ('Mid Prices') of the next step
            x_test = test[:, :-1]
            y_test = test[:, -1][:, -1] # Target is the last feature ('Mid Prices') of the next step

            # Reshape is needed by LSTM: [samples, time steps, features]
            # Shape check before reshape
            if x_train.shape[1] != seq_len or x_test.shape[1] != seq_len:
                print(f"Warning: Unexpected sequence length in train/test data. Expected {seq_len}.")
                # This might indicate an issue in the sequence generation loop


            # Reshape removed, Keras handles [samples, time steps, features] directly now for Dense after LSTM(return_sequences=False)
            # x_train = np.reshape(
            #     x_train, (x_train.shape[0], x_train.shape[1], amount_of_features)) # Already correct shape
            # x_test = np.reshape(
            #     x_test, (x_test.shape[0], x_test.shape[1], amount_of_features)) # Already correct shape


            return [x_train, y_train, x_test, y_test]
        # --- End Inner Helper Functions ---


        window = 5 # Sequence length
        amount_of_features = df_scaled.shape[1] # Should be 5 (O, C, L, H, Mid)

        load_result = load_data(df_scaled, window)

        # Check if load_data returned an error
        if load_result is None:
            print("Failed to load or prepare data sequences.")
            # Optionally delete the file if it's temporary
            # delete_stock_data(file_name)
            return None, None, None, None

        X_train, y_train, X_test, y_test = load_result

        # Check if data splits are empty
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
             print("Error: Training or testing data is empty after splitting.")
             # Optionally delete the file
             # delete_stock_data(file_name)
             return None, None, None, None


        # Prepare latest data point for prediction
        if len(df_scaled) < window:
             print(f"Error: Not enough data ({len(df_scaled)}) to form a sequence of length {window} for prediction.")
             # Optionally delete the file
             # delete_stock_data(file_name)
             return None, None, None, None

        x_latest_scaled = df_scaled[-window:] # Get the last 'window' points
        # Reshape for prediction: [1, time steps, features]
        x_latest_scaled = np.reshape(x_latest_scaled, (1, window, amount_of_features))


        # Build model using determined input shape
        # Input shape is (timesteps, features) -> (window, amount_of_features)
        model = build_model((window, amount_of_features))

        # Train the model
        model.fit(
            X_train,
            y_train,
            batch_size=512,
            epochs=20, # Consider making epochs configurable or using early stopping
            validation_split=0.1,
            verbose=1) # Set to 0 or 2 for less output

        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore, sqrt(trainScore)))
        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore, sqrt(testScore)))


        # Predictions
        p_scaled = model.predict(X_test)
        p_latest_scaled = model.predict(x_latest_scaled)

        # Inverse transform predictions and actual values
        # We need to reconstruct the scaled shape [samples, features] to use scaler.inverse_transform
        # Since we only predict the 'Mid Prices' (last column), create dummy arrays

        # Create dummy array for inverse scaling p_scaled
        dummy_p = np.zeros((len(p_scaled), amount_of_features))
        dummy_p[:, -1] = p_scaled.flatten() # Put prediction in the last column ('Mid Prices')
        p_inversed = scaler.inverse_transform(dummy_p)[:, -1] # Inverse transform and get the last column

        # Create dummy array for inverse scaling y_test
        dummy_y = np.zeros((len(y_test), amount_of_features))
        dummy_y[:, -1] = y_test.flatten() # Put actuals in the last column ('Mid Prices')
        y_test_inversed = scaler.inverse_transform(dummy_y)[:, -1] # Inverse transform and get the last column

        # Inverse transform the single latest prediction
        dummy_latest = np.zeros((1, amount_of_features))
        dummy_latest[:, -1] = p_latest_scaled.flatten()
        p_latest_inversed = scaler.inverse_transform(dummy_latest)[0, -1]


        # Calculate RMSE on the *inversed* (original scale) test data
        rmse = sqrt(mean_squared_error(y_test_inversed, p_inversed))
        print('Test RMSE (Original Scale): %.2f' % rmse)

        # Prepare return values
        p_list = p_inversed.tolist()
        y_test_list = y_test_inversed.tolist()
        tomorrow_prediction = p_latest_inversed

        # Optional: Delete the downloaded CSV file
        def delete_stock_data(fname):
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                    print(f"File '{fname}' removed successfully.")
                except OSError as e:
                    print(f"Error removing file '{fname}': {e}")
            else:
                print(f"File '{fname}' does not exist, cannot remove.")

        delete_stock_data(file_name) # Call the delete function


        return p_list, y_test_list, tomorrow_prediction, rmse


# --- Example Usage (kept commented out) ---
# Make sure TensorFlow/Keras and other dependencies are installed:
# pip install tensorflow numpy pandas scikit-learn

# ob = LSTM_Model()
# ticker_symbol = "IBM" # Example: Use IBM instead of TCS if needed
# results = ob.LSTM_Pred(ticker_symbol)

# if results and results[0] is not None: # Check if prediction was successful
#      p, y, tomorrow, rmse_val = results
#      print(f"Predictions on test set (original scale) for {ticker_symbol}:")
#      # print(p) # Uncomment to see all test predictions
#      print("...")
#      print(f"Actual values on test set (original scale) for {ticker_symbol}:")
#      # print(y) # Uncomment to see all test actuals
#      print("...")
#      print(f"\nTomorrow's predicted Mid Price for {ticker_symbol}: {tomorrow:.2f}")
#      print(f"RMSE on Test Set (original scale): {rmse_val:.2f}")
# else:
#      print(f"Could not get prediction for {ticker_symbol}.")