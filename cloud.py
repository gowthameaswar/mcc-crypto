import streamlit as st
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import streamlit as st

def main():
    # Add a radio button to switch between pages
    page = st.sidebar.radio("Select a page", ("Predictor", "View"))

    # Show the appropriate page based on the user's selection
    if page == "Predictor":
        predictor()
    elif page == "View":
        view()



def predictor():
    # Title and description
    st.title('Cryptocurrency Price Predictor')
    st.write('This app uses LSTM to predict cryptocurrency prices for the next 15 days.')

    # Get coin name input from user
    coin_name = st.text_input('Enter the name of the coin:', 'bitcoin')

    # Define the API endpoint and parameters
    url = f'https://api.coingecko.com/api/v3/coins/{coin_name}/market_chart'
    params = {'vs_currency': 'inr', 'days': '365'}

    # Make a request to the API endpoint and convert the response to JSON
    response = requests.get(url, params=params)
    data = response.json()

    # Extract the prices from the response and convert to a Pandas DataFrame
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    prices.set_index('timestamp', inplace=True)

    # Define the window size for the LSTM model
    window_size = st.number_input('Choose a window size for the LSTM model:', value=165, min_value=1, max_value=365, step=1)

    # Define a function to create the input and output sequences
    def create_sequences(data, window_size):
        X = []
        y = []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    # Create the input and output sequences for the LSTM model
    X, y = create_sequences(prices.values, window_size)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Normalize the data
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, 1)))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # Evaluate the model on the testing set
    loss = model.evaluate(X_test, y_test, verbose=0)
    st.write('Test loss:', loss)

    # Generate predictions for the next 7 days
    future_days = st.number_input('Choose the number of days to predict for:', value=15, min_value=1, max_value=30, step=1)
    last_sequence = X[-1]
    predictions = []
    for i in range(future_days):
        next_sequence = model.predict(last_sequence.reshape(1, window_size, 1))
        predictions.append(next_sequence[0, 0])
        last_sequence = np.append(last_sequence[1:], next_sequence[0])

    # Denormalize the predictions
    predictions = np.array(predictions) * y_std + y_mean

    # Print the predictions
    st.write('Predictions for', coin_name + ':')
    for i in range(future_days):
        st.write(prices.index[-1] + pd.DateOffset(days=i+1), predictions[i])

    import plotly.graph_objs as go

    # Create a line chart of historical prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index.tolist(), y=prices['price'].tolist(), name='Historical Prices'))
    fig.update_layout(title='Historical Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Create a line chart of predicted prices
    fig = go.Figure()
    future_dates = pd.date_range(start=prices.index[-1], periods=future_days+1, closed='right')[1:]
    fig.add_trace(go.Scatter(x=future_dates.tolist(), y=predictions.tolist(), name='Predicted Prices'))
    fig.update_layout(title='Predicted Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)



    # Create a bar chart of market caps
    market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
    market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
    market_caps.set_index('timestamp', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=market_caps.index.tolist(), y=market_caps['market_cap'].tolist(), name='Market Cap'))
    fig.update_layout(title='Market Cap', xaxis_title='Date', yaxis_title='Market Cap')
    st.plotly_chart(fig)

    # Create a bar chart of total volumes
    total_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    total_volumes['timestamp'] = pd.to_datetime(total_volumes['timestamp'], unit='ms')
    total_volumes.set_index('timestamp', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=total_volumes.index.tolist(), y=total_volumes['volume'].tolist(), name='Volume'))
    fig.update_layout(title='Total Volume', xaxis_title='Date', yaxis_title='Volume')
    st.plotly_chart(fig)

def view():
    # Set page title

    # Create input field for coin name
    coin_name = st.text_input("Enter the name of the cryptocurrency")

    if coin_name:
        # Send a request to the CoinGecko API to get the data
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=inr&ids={coin_name}&order=market_cap_desc&per_page=100&page=1&sparkline=false&locale=en"
        response = requests.get(url)

        # Parse the data and display the values
        data = response.json()[0]
        price = data["current_price"]
        high_price = data["high_24h"]
        low_price = data["low_24h"]

        st.write(f"Current Price: {price}")
        st.write(f"24-hour High: {high_price}")
        st.write(f"24-hour Low: {low_price}")


if __name__ == "__main__":
    main()
