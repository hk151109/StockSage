import numpy as np
import pandas as pd
from dotenv import load_dotenv
import itertools
from tensorflow.keras.layers import Dense, Activation, Dropout, Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time
import datetime as dt
import urllib.request
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import ta
import matplotlib.pyplot as plt
from datetime import timedelta
import itertools

load_dotenv() #load the env variables

class Enhanced_LSTM_Model:

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
            raise ValueError("No Alpha Vantage API keys configured or loaded.")

        # Get the key at the current index
        key = cls.api_keys[cls.current_key_index]

        # Move to the next index, wrapping around using modulo
        cls.current_key_index = (cls.current_key_index + 1) % len(cls.api_keys)

        # Optional: Print which key index is being used (for debugging)
        print(f"Using API key index: {(cls.current_key_index - 1 + len(cls.api_keys)) % len(cls.api_keys)}")

        return key

    @staticmethod
    def add_technical_indicators(df):
        """Add technical indicators to the dataframe."""
        # Make sure we have enough data for calculating indicators
        if len(df) < 30:
            print("Warning: Not enough data for accurate technical indicators")
            
        # Initialize an object of the Indicators class with the dataframe
        # Calculate RSI (Relative Strength Index)
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # Calculate MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
        
        # Calculate Simple Moving Averages
        df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        
        # Add Volatility (Standard Deviation of returns)
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Add momentum indicators
        df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=12).roc()
        
        # Forward fill NaN values that occur at the beginning due to lookback windows
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        # If any NaNs still exist, fill with zeros (should be minimal at this point)
        df.fillna(0, inplace=True)
        
        return df

    @staticmethod
    def time_series_split(data, n_splits=5):
        """Generate indices to split data for time series cross-validation."""
        total_samples = len(data)
        indices = np.arange(total_samples)
        
        # Calculate the minimum training size
        test_size = total_samples // (n_splits + 1)
        train_size = total_samples - n_splits * test_size
        
        # Generate the splits
        splits = []
        for i in range(n_splits):
            start = i * test_size
            stop = start + train_size + (i * test_size)
            train_indices = indices[start:stop - test_size]
            test_indices = indices[stop - test_size:stop]
            
            splits.append((train_indices, test_indices))
            
        return splits

    @staticmethod
    def build_model(input_shape_tuple, config=None):
        """Build LSTM model with configurable parameters."""
        if config is None:
            config = {
                'lstm_units_1': 50,
                'lstm_units_2': 100,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'bidirectional': False
            }
        
        model = Sequential()
        
        # First LSTM layer
        if config['bidirectional']:
            model.add(Bidirectional(
                LSTM(
                    units=config['lstm_units_1'],
                    return_sequences=True),
                input_shape=input_shape_tuple))
        else:
            model.add(LSTM(
                input_shape=input_shape_tuple,
                units=config['lstm_units_1'],
                return_sequences=True))
        
        model.add(Dropout(config['dropout_rate']))
        
        # Second LSTM layer
        if config['bidirectional']:
            model.add(Bidirectional(
                LSTM(
                    units=config['lstm_units_2'],
                    return_sequences=False)))
        else:
            model.add(LSTM(
                config['lstm_units_2'],
                return_sequences=False))
        
        model.add(Dropout(config['dropout_rate']))
        
        # Output layer
        model.add(Dense(units=1))
        model.add(Activation('linear'))
        
        start = time.time()
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        print('Compilation time : ', time.time() - start)
        
        return model

    @staticmethod
    def load_data(stock_scaled, seq_len):
        """Prepare sequences from scaled data."""
        amount_of_features = stock_scaled.shape[1]
        data = stock_scaled
        sequence_length = seq_len + 1
        result = []
        
        # Ensure index range is valid
        if len(data) <= sequence_length:
            print("Error: Data length is too short for the sequence length.")
            return None

        for index in range(len(data) - sequence_length + 1):
            result.append(data[index: index + sequence_length])

        result = np.array(result)
        if result.shape[0] == 0:
             print("Error: No sequences generated. Check data length and sequence length.")
             return None

        row = round(0.8 * result.shape[0])  # Changed from 0.75 to 0.8
        if row == 0 or row == result.shape[0]:
            print("Warning: Train/Test split resulted in empty set. Adjusting split or check data.")
            row = max(1, int(result.shape[0] * 0.8)) if result.shape[0] > 1 else 0

        train = result[:int(row), :]
        test = result[int(row):, :]

        if train.shape[0] == 0 or test.shape[0] == 0:
             print(f"Error: Train ({train.shape[0]}) or Test ({test.shape[0]}) set is empty.")
             return None

        x_train = train[:, :-1]
        y_train = train[:, -1][:, -1]  # Target is the last feature of the next step
        x_test = test[:, :-1]
        y_test = test[:, -1][:, -1]  # Target is the last feature of the next step

        # Shape check before reshape
        if x_train.shape[1] != seq_len or x_test.shape[1] != seq_len:
            print(f"Warning: Unexpected sequence length in train/test data. Expected {seq_len}.")

        return [x_train, y_train, x_test, y_test]

@classmethod
def hyperparameter_tuning(cls, X_train, y_train, X_val, y_val, input_shape):
    """Perform grid search for hyperparameter tuning."""
    print("Starting hyperparameter tuning...")
    
    # Define hyperparameter grid
    param_grid = {
        'lstm_units_1': [32, 64, 128],
        'lstm_units_2': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005],
        'bidirectional': [False, True]
    }
    
    # Use a subset of combinations to avoid exponential growth
    keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in keys]
    
    # Limit to a reasonable number of combinations
    max_combinations = 8  # Adjust based on your computational resources
    all_combinations = list(itertools.product(*param_values))
    
    # Sample combinations if there are too many
    if len(all_combinations) > max_combinations:
        # Randomly sample or take first few combinations
        selected_combinations = all_combinations[:max_combinations]
    else:
        selected_combinations = all_combinations
    
    best_val_loss = float('inf')
    best_config = None
    best_model = None
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"Expected input shape: {input_shape}")
    
    for i, combination in enumerate(selected_combinations):
        config = {keys[j]: combination[j] for j in range(len(keys))}
        print(f"\nTrying combination {i+1}/{len(selected_combinations)}: {config}")
        
        # Build model with this configuration
        model = cls.build_model(input_shape, config)
        
        # Define callbacks for this training run
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train with early stopping
        # Make sure X_train is 3D: [samples, time steps, features]
        if len(X_train.shape) != 3:
            print(f"Reshaping X_train from {X_train.shape} to match expected input shape...")
            if len(X_train.shape) == 2 and X_train.shape[1] == input_shape[0] * input_shape[1]:
                # If flattened, reshape back to 3D
                X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1])
            else:
                print("ERROR: Cannot reshape X_train to expected dimensions")
                continue
                
        # Do the same for X_val
        if len(X_val.shape) != 3:
            print(f"Reshaping X_val from {X_val.shape} to match expected input shape...")
            if len(X_val.shape) == 2 and X_val.shape[1] == input_shape[0] * input_shape[1]:
                # If flattened, reshape back to 3D
                X_val = X_val.reshape(X_val.shape[0], input_shape[0], input_shape[1])
            else:
                print("ERROR: Cannot reshape X_val to expected dimensions")
                continue
        
        print(f"After reshaping: X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        
        try:
            history = model.fit(
                X_train, y_train,
                batch_size=32,  # Smaller batch size for tuning
                epochs=15,  # Fewer epochs for tuning
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1  # Changed to 1 for more feedback
            )
            
            # Evaluate on validation data
            val_loss = min(history.history['val_loss'])
            print(f"Validation loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
                best_model = model
                print(f"New best configuration found!")
        except Exception as e:
            print(f"Error during training: {e}")
            continue
    
    if best_config is None:
        print("WARNING: No successful configurations found. Using default config.")
        best_config = {
            'lstm_units_1': 50,
            'lstm_units_2': 100,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'bidirectional': False
        }
        best_model = cls.build_model(input_shape, best_config)
        
        # Try to train with default config
        try:
            history = best_model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=10,
                validation_data=(X_val, y_val),
                verbose=1
            )
        except Exception as e:
            print(f"Error training with default config: {e}")
            # Return the untrained model as a last resort
    
    print("\nBest hyperparameters:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return best_model, best_config

    @staticmethod
    def evaluate_model(y_true, y_pred):
        """Evaluate model with multiple metrics."""
        metrics = {}
        
        # Root Mean Squared Error
        metrics['rmse'] = sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Directional Accuracy
        direction_true = np.diff(y_true)
        direction_pred = np.diff(y_pred)
        direction_accuracy = np.mean((direction_true > 0) == (direction_pred > 0))
        metrics['direction_accuracy'] = direction_accuracy * 100
        
        return metrics

    @staticmethod
    def backtest_strategy(prices, predictions, initial_capital=10000.0, commission=0.001):
        """Backtest a simple trading strategy based on predictions."""
        capital = initial_capital
        position = 0  # 0 = cash, 1 = holding stock
        trades = []
        portfolio_values = [capital]
        
        # Convert to numpy arrays if they're not already
        prices = np.array(prices)
        predictions = np.array(predictions)
        
        for i in range(1, len(predictions)):
            # Calculate signals: buy if predicted price increase, sell if predicted decrease
            if predictions[i] > prices[i-1] and position == 0:
                # Buy signal
                shares = capital / prices[i-1]
                cost = shares * prices[i-1]
                commission_cost = cost * commission
                shares = shares * (1 - commission)  # Reduce shares by commission
                
                trades.append({
                    'type': 'buy',
                    'price': prices[i-1],
                    'shares': shares,
                    'value': cost,
                    'commission': commission_cost,
                    'capital_remaining': 0
                })
                
                position = 1
                capital = 0  # All capital used to buy shares
                
            elif predictions[i] < prices[i-1] and position == 1:
                # Sell signal
                value = shares * prices[i-1]
                commission_cost = value * commission
                capital = value * (1 - commission)  # Reduce capital by commission
                
                trades.append({
                    'type': 'sell',
                    'price': prices[i-1],
                    'shares': shares,
                    'value': value,
                    'commission': commission_cost,
                    'capital_remaining': capital
                })
                
                position = 0
                shares = 0
            
            # Calculate portfolio value
            if position == 0:
                portfolio_values.append(capital)
            else:
                portfolio_values.append(shares * prices[i-1])
        
        # Close position at the end if still holding
        if position == 1:
            value = shares * prices[-1]
            commission_cost = value * commission
            capital = value * (1 - commission)
            
            trades.append({
                'type': 'sell',
                'price': prices[-1],
                'shares': shares,
                'value': value,
                'commission': commission_cost,
                'capital_remaining': capital
            })
            
            portfolio_values[-1] = capital
        
        # Calculate performance metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate buy & hold return for comparison
        buy_hold_return = (prices[-1] - prices[0]) / prices[0] * 100
        
        # Calculate max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Number of trades
        num_trades = len(trades)
        
        # Calculate Sharpe ratio (simplified, assuming zero risk-free rate)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'max_drawdown_pct': max_drawdown,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_values': portfolio_values,
            'trades': trades
        }
        
        return results

    @classmethod
    def LSTM_Pred(cls, tick, window_size=10, tune_hyperparams=True, do_backtesting=True):
        """Enhanced LSTM prediction with configurable window size and hyperparameter tuning."""
        data_source = 'alphavantage'

        if data_source == 'alphavantage':
            # ====================== Loading Data from Alpha Vantage ==================================
            try:
                api_key = cls._get_next_api_key()
            except ValueError as e:
                print(f"API Key Error: {e}")
                return None

            ticker = tick
            url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
            file_to_save = f'stock_market_data-{ticker}.csv'

            if not os.path.exists(file_to_save):
                try:
                    with urllib.request.urlopen(url_string) as url:
                        data = json.loads(url.read().decode())
                        # extract stock market data
                        if 'Time Series (Daily)' not in data:
                             print(f"Error fetching data from Alpha Vantage for {ticker}.")
                             print(f"Response: {data}")
                             return None

                        data = data['Time Series (Daily)']
                        df = pd.DataFrame(
                            columns=['Date', 'Low', 'High', 'Close', 'Open', 'Volume'])
                        
                        rows_list = []
                        for k, v in data.items():
                            date = dt.datetime.strptime(k, '%Y-%m-%d')
                            # Check if all expected keys are present
                            if all(key in v for key in ['3. low', '2. high', '4. close', '1. open', '5. volume']):
                                data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                                          float(v['4. close']), float(v['1. open']), float(v['5. volume'])]
                                rows_list.append(data_row)
                            else:
                                print(f"Warning: Missing data for date {k}. Skipping this entry.")

                        temp_df = pd.DataFrame(rows_list, columns=['Date', 'Low', 'High', 'Close', 'Open', 'Volume'])
                        df = pd.concat([df, temp_df], ignore_index=True)

                    print('Data saved to : %s' % file_to_save)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date')
                    df.to_csv(file_to_save, index=False)

                except Exception as e:
                    print(f"An error occurred during data fetching: {e}")
                    return None
            else:
                print('File already exists. Loading data from CSV')
                df = pd.read_csv(file_to_save, parse_dates=['Date'])

        file_name = f'stock_market_data-{ticker}.csv'
        if not os.path.exists(file_name):
             print(f"CSV file {file_name} not found. Cannot proceed.")
             return None

        df = pd.read_csv(file_name, parse_dates=['Date'])
        df = df.sort_values('Date')

        # Add technical indicators
        df = cls.add_technical_indicators(df)
        
        # Create mid price
        high_prices = df.loc[:, 'High']
        low_prices = df.loc[:, 'Low']
        df["Mid Price"] = (high_prices + low_prices) / 2.0

        # Store dates for reference and plotting
        dates = df['Date']
        df.drop("Date", axis=1, inplace=True)

        # Define the target column (Mid Price)
        target_col = "Mid Price"
        
        # Ensure target column is the last column for easier processing
        cols = df.columns.tolist()
        cols.remove(target_col)
        cols.append(target_col)
        df = df[cols]

        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)

        # Scale the data
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df.values)
        
        # ================ Time Series Cross-Validation ================
        cv_splits = cls.time_series_split(df_scaled, n_splits=5)
        
        cv_metrics = []
        predictions_all = []
        actual_all = []
        dates_all = []
        
        # Get shape information for model building
        amount_of_features = df_scaled.shape[1]
        input_shape = (window_size, amount_of_features)
        
        # Placeholder for the best model across CV folds
        best_model = None
        best_config = None
        
        print(f"Running {len(cv_splits)} fold time-series cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            print(f"\nFold {fold+1}/{len(cv_splits)}")
            
            # Split data for this fold
            data_train = df_scaled[train_idx]
            data_test = df_scaled[test_idx]
            
            # Further split training data for validation (for hyperparameter tuning)
            val_size = int(len(data_train) * 0.2)
            train_data = data_train[:-val_size]
            val_data = data_train[-val_size:]
            
            # Prepare sequences
            X_train_seq = []
            y_train_seq = []
            for i in range(len(train_data) - window_size):
                X_train_seq.append(train_data[i:i+window_size])
                y_train_seq.append(train_data[i+window_size][-1])  # Target is the last column
            
            X_train_seq = np.array(X_train_seq)
            y_train_seq = np.array(y_train_seq)
            
            # Prepare validation sequences
            X_val_seq = []
            y_val_seq = []
            for i in range(len(val_data) - window_size):
                X_val_seq.append(val_data[i:i+window_size])
                y_val_seq.append(val_data[i+window_size][-1])
            
            X_val_seq = np.array(X_val_seq)
            y_val_seq = np.array(y_val_seq)
            
            # Prepare test sequences
            X_test_seq = []
            y_test_seq = []
            test_dates = []
            for i in range(len(data_test) - window_size):
                X_test_seq.append(data_test[i:i+window_size])
                y_test_seq.append(data_test[i+window_size][-1])
                test_dates.append(dates.iloc[test_idx[i+window_size]])
            
            X_test_seq = np.array(X_test_seq)
            y_test_seq = np.array(y_test_seq)
            
            # Hyperparameter tuning if enabled
            if tune_hyperparams and fold == 0:  # Only tune on first fold to save time
                best_model, best_config = cls.hyperparameter_tuning(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape
                )
            else:
                # Use default or best config from first fold
                if best_config is None:
                    # Default configuration if no tuning
                    best_config = {
                        'lstm_units_1': 64,
                        'lstm_units_2': 128,
                        'dropout_rate': 0.2,
                        'learning_rate': 0.001,
                        'bidirectional': True
                    }
                
                best_model = cls.build_model(input_shape, best_config)
            
            # Define callbacks for model training
            model_path = f'model_checkpoint_{ticker}_fold{fold}.h5'
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
                ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Train model with early stopping and learning rate reduction
            history = best_model.fit(
                X_train_seq, y_train_seq,
                batch_size=64,
                epochs=50,
                validation_data=(X_val_seq, y_val_seq),
                callbacks=callbacks,
                verbose=1
            )
            
            # Load the best model saved during training
            if os.path.exists(model_path):
                best_model = load_model(model_path)
                # Remove the temporary model file
                os.remove(model_path)
            
            # Make predictions
            y_pred_scaled = best_model.predict(X_test_seq)
            
            # Prepare for inverse scaling
            dummy_array = np.zeros((len(y_pred_scaled), amount_of_features))
            dummy_array[:, -1] = y_pred_scaled.flatten()
            
            # Inverse transform
            y_pred = scaler.inverse_transform(dummy_array)[:, -1]
            
            # Same for actuals
            dummy_array = np.zeros((len(y_test_seq), amount_of_features))
            dummy_array[:, -1] = y_test_seq
            y_test = scaler.inverse_transform(dummy_array)[:, -1]
            
            # Calculate metrics
            metrics = cls.evaluate_model(y_test, y_pred)
            metrics['fold'] = fold + 1
            cv_metrics.append(metrics)
            
            # Store results for this fold
            predictions_all.extend(y_pred)
            actual_all.extend(y_test)
            dates_all.extend(test_dates)
            
            print(f"Fold {fold+1} Results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        # Aggregate cross-validation metrics
        cv_summary = {metric: np.mean([fold[metric] for fold in cv_metrics]) for metric in cv_metrics[0] if metric != 'fold'}
        print("\nCross-Validation Summary:")
        for metric_name, value in cv_summary.items():
            print(f"Mean {metric_name}: {value:.4f}")
        
        # Run backtesting if enabled
        backtest_results = None
        if do_backtesting and len(predictions_all) > 0:
            print("\nBacktesting trading strategy...")
            backtest_results = cls.backtest_strategy(actual_all, predictions_all)
            
            print(f"Strategy Return: {backtest_results['total_return_pct']:.2f}%")
            print(f"Buy & Hold Return: {backtest_results['buy_hold_return_pct']:.2f}%")
            print(f"Number of Trades: {backtest_results['num_trades']}")
            print(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
        
        # Make prediction for tomorrow
        latest_data = df_scaled[-window_size:]
        x_latest = np.reshape(latest_data, (1, window_size, amount_of_features))
        
        # Use the last trained model for prediction
        p_latest_scaled = best_model.predict(x_latest)
        
        # Inverse transform
        dummy_latest = np.zeros((1, amount_of_features))
        dummy_latest[:, -1] = p_latest_scaled.flatten()
        p_latest = scaler.inverse_transform(dummy_latest)[0, -1]
        
        # Prepare tomorrow's date
        last_date = dates.iloc[-1]
        next_business_day = last_date + timedelta(days=1)
        while next_business_day.weekday() > 4:  # Skip weekends
            next_business_day += timedelta(days=1)
        
        print(f"\nPredicted {target_col} for {ticker} on {next_business_day.strftime('%Y-%m-%d')}: {p_latest:.2f}")
        
        # Create a plot of actual vs predicted prices
        plt.figure(figsize=(12, 6))
        
        # Convert dates_all to a format suitable for plotting
        plot_dates = [date.date() for date in dates_all]
        
        plt.plot(plot_dates, actual_all, label='Actual', color='blue')
        plt.plot(plot_dates, predictions_all, label='Predicted', color='red', linestyle='--')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel(f'{target_col}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_file = f'{ticker}_prediction_plot.png'
        plt.savefig(plot_file)
        plt.close()
        print(f"Prediction plot saved to {plot_file}")
        
        # Optional: Delete the downloaded CSV file
        # Commented out to allow for reuse in future runs
        # if os.path.exists(file_name):
        #     os.remove(file_name)
        #     print(f"File '{file_name}' removed successfully.")
        
        return {
            'predictions': predictions_all,
            'actuals': actual_all,
            'dates': dates_all,
            'tomorrow_prediction': p_latest,
            'tomorrow_date': next_business_day,
            'cv_metrics': cv_metrics,
            'cv_summary': cv_summary,
            'backtest_results': backtest_results,
            'best_model': best_model,
            'best_config': best_config,
            'scaler': scaler,
            'window_size': window_size
        }


# Example Usage
if __name__ == "__main__":
    # Initialize model
    model = Enhanced_LSTM_Model()
    
    # Set parameters
    ticker_symbol = "AAPL"
    window_size = 15  # Configurable window size (sequence length)
    
    # Run the model with hyperparameter tuning and backtesting
    results = model.LSTM_Pred(
        tick=ticker_symbol, 
        window_size=window_size, 
        tune_hyperparams=True,
        do_backtesting=True
    )
    
    if results:
        # Print summary metrics
        print("\n----- Summary Results -----")
        print(f"CV Summary Metrics:")
        for metric, value in results['cv_summary'].items():
            print(f"  {metric}: {value:.4f}")
        
        if results['backtest_results']:
            print(f"\nBacktest Results:")
            print(f"  Initial Capital: ${results['backtest_results']['initial_capital']:.2f}")
            print(f"  Final Value: ${results['backtest_results']['final_value']:.2f}")
            print(f"  Total Return: {results['backtest_results']['total_return_pct']:.2f}%")
            print(f"  Buy & Hold Return: {results['backtest_results']['buy_hold_return_pct']:.2f}%")
            print(f"  Number of Trades: {results['backtest_results']['num_trades']}")
            print(f"  Max Drawdown: {results['backtest_results']['max_drawdown_pct']:.2f}%")
            print(f"  Sharpe Ratio: {results['backtest_results']['sharpe_ratio']:.4f}")
        
        print(f"\nNext Trading Day Prediction ({results['tomorrow_date'].strftime('%Y-%m-%d')}):")
        print(f"  Predicted Mid Price: ${results['tomorrow_prediction']:.2f}")
        
        # You can save the model for future use
        if results['best_model']:
            model_save_path = f"{ticker_symbol}_lstm_model.h5"
            results['best_model'].save(model_save_path)
            print(f"\nModel saved to {model_save_path}")
            
            # Save configuration
            import json
            config_save_path = f"{ticker_symbol}_model_config.json"
            with open(config_save_path, 'w') as f:
                # Convert numpy values to native Python types for JSON serialization
                config_to_save = {k: (float(v) if isinstance(v, np.floating) else v) 
                                 for k, v in results['best_config'].items()}
                json.dump(config_to_save, f, indent=4)
            print(f"Model configuration saved to {config_save_path}")
    else:
        print("Failed to run model. Check error messages above.")