import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tabulate import tabulate

# The openpyxl module is required for Excel saving
try:
    import openpyxl
except ImportError:
    print("Error: 'openpyxl' module is missing. Please run 'pip install openpyxl'.")

# The 50 most popular coins listed on Binance (using USD pairs for yfinance)
def get_popular_coins():
    return [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "SOL-USD", "MATIC-USD",
        "DOT-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "UNI-USD", "LINK-USD", "XMR-USD", "ATOM-USD", "ETC-USD",
        "XLM-USD", "BCH-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "MANA-USD", "SAND-USD", "AXS-USD", "THETA-USD",
        "XTZ-USD", "FTM-USD", "HBAR-USD", "EGLD-USD", "HNT-USD", "GRT-USD", "KSM-USD", "NEAR-USD", "CHZ-USD", "ENJ-USD",
        "STX-USD", "ZIL-USD", "DASH-USD", "CRV-USD", "COMP-USD", "YFI-USD", "1INCH-USD", "BAT-USD", "LRC-USD", "REN-USD",
        "QTUM-USD", "ZRX-USD", "MKR-USD", "OMG-USD", "DGB-USD", "NANO-USD", "HBAR-USD", "WAVES-USD", "ICX-USD", "DCR-USD",
        "LEND-USD", "SUSHI-USD", "BAT-USD", "REN-USD", "CVC-USD", "STPT-USD", "MATIC-USD", "SAND-USD", "LRC-USD", "FET-USD",
        "STMX-USD", "COTI-USD", "CTXC-USD", "MANA-USD", "FLOKI-USD", "HNT-USD", "LDO-USD", "SAND-USD", "LTC-USD", "XEM-USD",
    ]

# Function that fetches price and market cap data using the Yahoo Finance API
def fetch_coin_data(coin_symbol, retry_count=3):
    try:
        coin = yf.Ticker(coin_symbol)
        hist = coin.history(period="180d")
        if hist.empty:
            print(f"Error: No valid price data found for {coin_symbol}.")
            return None, "Data Not Available"
        
        df = hist.reset_index()[["Date", "Close"]]
        df.rename(columns={"Date": "timestamp"}, inplace=True)
        
        # Market cap data is not consistently available via the Yahoo Finance API history,
        # so we return "Data Not Available"
        market_cap = "Data Not Available"
        
        return df, market_cap
    except Exception as e:
        print(f"An error occurred during API request: {e}")
        return None, "Data Not Available"

# Function that scales the data and prepares it for the model, considering past prices
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i])
        y.append(data_scaled[i, 0])
    
    return np.array(X), np.array(y), scaler

# Function that creates the LSTM model
def create_model(X_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to determine the time to reach the predicted value
def determine_prediction_duration(data, predicted_value):
    data['Close_Diff'] = data['Close'].diff()
    average_daily_change = data['Close_Diff'].abs().mean()
    remaining_difference = abs(predicted_value - data['Close'].iloc[-1])
    # Avoid division by zero if daily change is zero (unlikely for price data)
    if average_daily_change == 0:
        return 9999 # Return a large number if no change is observed
        
    estimated_day_count = remaining_difference / average_daily_change
    return int(round(estimated_day_count))

# Function that calculates the future price and estimated time for a specific coin
def predict_price(coin_symbol, seed=42):
    data, market_cap = fetch_coin_data(coin_symbol)
    if data is None or data.empty:
        return None
    
    X, y, scaler = prepare_data(data)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    np.random.seed(seed)  # Control randomness
    model = create_model(X_train)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    
    # Predict and inverse transform
    y_pred = model.predict(X_test, verbose=0)
    # The scaler was fitted on a single feature (Close), but inverse_transform expects the original shape (1 feature for the Close price, and an empty column for MarketCap which we are ignoring/setting to 0)
    y_pred = scaler.inverse_transform(np.column_stack((y_pred, np.zeros((len(y_pred), 1)))))[:, 0]
    y_real = scaler.inverse_transform(np.column_stack((y_test, np.zeros((len(y_test), 1)))))[:, 0]
    
    predicted_duration = determine_prediction_duration(data, y_pred[-1])
    print(f"Real value: {y_real[-1]:.4f}, Predicted value: {y_pred[-1]:.4f}, Estimated duration: {predicted_duration} days")
    return y_pred[-1], predicted_duration

# Function that takes the coin symbol from the user and performs analysis
def user_coin_analysis():
    while True:
        coins = get_popular_coins()
        # Format the list of coins for display
        coin_list_display = [[f"{i + j * 8 + 1}: {coins[i + j * 8]}" if i + j * 8 < len(coins) else "" for i in range(8)] for j in range((len(coins) + 7) // 8)]
        print(tabulate(coin_list_display, tablefmt="grid"))
        
        while True:
            try:
                choice = int(input("Please enter the number of the coin you want to analyze (0 to cancel): "))
                if choice == 0:
                    return None
                if 1 <= choice <= len(coins):
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        coin_symbol = coins[choice - 1]
        result = predict_price(coin_symbol)
        
        if result:
            prediction, predicted_duration = result
            data, market_cap = fetch_coin_data(coin_symbol)
            if data is not None and not data.empty:
                current_price = data["Close"].iloc[-1]
                percentage_change = ((prediction - current_price) / current_price) * 100
                print(f"{coin_symbol} - Current Price: {current_price:.4f}, Predicted Price: {prediction:.4f}, Change %: {percentage_change:.2f}, Estimated Duration: {predicted_duration} days")
                return {coin_symbol: {"Current Price": current_price, "Predicted Price": prediction, "Change %": percentage_change, "Estimated Duration (days)": predicted_duration, "Market Cap": market_cap}}
            else:
                print(f"Analysis could not be performed for {coin_symbol}.")
        else:
            print(f"Analysis could not be performed for {coin_symbol}.")
        
        while True:
            repeat = input("Do you want to analyze again? (y/n): ")
            if repeat.lower() not in ('y', 'n'):
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            break
        if repeat.lower() != 'y':
            break
    return None

# Function that saves the results to the desktop in Excel format
def save_results(results):
    if not results:
        print("No results to save.")
        return
    
    timestamp = datetime.datetime.now().strftime("%d%m%y_%H%M")
    filename = f"prediction_results_{timestamp}.xlsx"
    # Create the 'analysis' folder on the desktop if it doesn't exist
    desktop_analysis_path = os.path.join(os.path.expanduser("~"), "Desktop", "analysis")
    os.makedirs(desktop_analysis_path, exist_ok=True)
    
    file_path = os.path.join(desktop_analysis_path, filename)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = "Coin"
    try:
        df.to_excel(file_path, index=True, engine='openpyxl')
        print(f"Results saved to: {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the Excel file: {e}")

# Example usage of the function
if __name__ == "__main__":
    while True:
        results = user_coin_analysis()
        if results:
            save_results(results)

        stop_program = input("Do you want to exit the program? (y/n): ")
        if stop_program.lower() != 'n':
            break