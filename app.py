import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objs as go
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# API Keys
EXCHANGE_API_KEY = "E8M2IEQYPFF6V7L1"
NEWS_API_KEY = "ff5a86e2cb1642fa92f7186aa1cdaaf9"

# Currency List
CURRENCIES = ["USD", "EUR", "GBP", "INR", "JPY", "AUD", "CAD", "CHF", "CNY"]

# TradingView Symbol Helper
TRADINGVIEW_SYMBOLS = {
    "USDINR": "FX_IDC:USDINR",
    "EURUSD": "FX_IDC:EURUSD",
    "USDJPY": "FX_IDC:USDJPY",
    "GBPUSD": "FX_IDC:GBPUSD",
    "AUDUSD": "FX_IDC:AUDUSD",
    "USDCAD": "FX_IDC:USDCAD",
    "USDCHF": "FX_IDC:USDCHF",
    "USDCNY": "FX_IDC:USDCNY",
    "EURINR": "FX_IDC:EURINR",
    "GBPINR": "FX_IDC:GBPINR",
}

def fetch_exchange_rate_data(from_currency, to_currency):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_currency,
        "to_symbol": to_currency,
        "apikey": EXCHANGE_API_KEY,
        "outputsize": "full",
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "Time Series FX (Daily)" in data:
        raw_data = data["Time Series FX (Daily)"]
        df = pd.DataFrame(raw_data).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    else:
        return None

def fetch_sentiment_score(currency):
    today = datetime.datetime.today().date()
    url = f"https://newsapi.org/v2/everything?q={currency}+economy+geopolitics&from={today}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    sentiment_total = 0
    count = 0
    for article in articles[:10]:
        analysis = TextBlob(article.get("title", "") + " " + article.get("description", ""))
        sentiment_total += analysis.sentiment.polarity
        count += 1
    return sentiment_total / max(count, 1)

def adjust_with_sentiment(predictions, sentiment):
    adjustment = 1 + (sentiment * 0.05)
    return predictions * adjustment

def predict_lstm(data, days):
    series = data['4. close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)
    X, y = [], []
    time_steps = 10
    for i in range(time_steps, len(scaled_series)):
        X.append(scaled_series[i - time_steps:i])
        y.append(scaled_series[i])
    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(Input(shape=(time_steps, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    predictions = []
    current_input = scaled_series[-time_steps:].reshape(1, time_steps, 1)
    for _ in range(int(days)):
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0][0])
        current_input = np.concatenate([current_input[:, 1:, :], next_pred.reshape(1, 1, 1)], axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def predict_arima(series, days):
    model = ARIMA(series, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=days)
    return forecast

def predict_ets(series, days):
    model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
    forecast = model.forecast(steps=days)
    return forecast

def backtest_models(data):
    base_series = data['4. close'][-60:]
    actual = base_series[-7:].values
    lstm_back = predict_lstm(data.iloc[:-7], 7)
    arima_back = predict_arima(base_series[:-7], 7)
    ets_back = predict_ets(base_series[:-7], 7)
    scores = {
        'LSTM': {
            'RMSE': np.sqrt(mean_squared_error(actual, lstm_back)),
            'MAE': mean_absolute_error(actual, lstm_back)
        },
        'ARIMA': {
            'RMSE': np.sqrt(mean_squared_error(actual, arima_back)),
            'MAE': mean_absolute_error(actual, arima_back)
        },
        'ETS': {
            'RMSE': np.sqrt(mean_squared_error(actual, ets_back)),
            'MAE': mean_absolute_error(actual, ets_back)
        }
    }
    return scores

st.set_page_config(page_title="Currency Exchange Forecast", layout="wide")
st.title("💱 Currency Exchange Rate Forecast")

col1, col2, col3 = st.columns(3)
with col1:
    from_currency = st.selectbox("From Currency", CURRENCIES, index=0)
with col2:
    to_currency = st.selectbox("To Currency", CURRENCIES, index=3)
with col3:
    model_choice = st.selectbox("Select Model", ["LSTM", "ARIMA", "ETS"])

days = st.slider("Number of Days to Predict", min_value=1, max_value=365, value=7)

if st.button("🔍 Predict"):
    data = fetch_exchange_rate_data(from_currency, to_currency)
    if data is None:
        st.error("❌ Failed to fetch exchange rate data.")
    else:
        sentiment = fetch_sentiment_score(to_currency)
        lstm_pred = predict_lstm(data, days)
        arima_pred = predict_arima(data['4. close'], days)
        ets_pred = predict_ets(data['4. close'], days)
        lstm_adj = adjust_with_sentiment(lstm_pred, sentiment)
        arima_adj = adjust_with_sentiment(arima_pred, sentiment)
        ets_adj = adjust_with_sentiment(ets_pred, sentiment)

        if model_choice == "LSTM":
            selected_preds = lstm_adj
        elif model_choice == "ARIMA":
            selected_preds = arima_adj
        else:
            selected_preds = ets_adj

        future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=days)
        prediction_df = pd.DataFrame({"Date": future_dates, "Prediction": selected_preds})

        st.subheader(f"📈 Forecasted Exchange Rates using {model_choice}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[-60:], y=data['4. close'].tail(60), name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=prediction_df["Date"], y=prediction_df["Prediction"], name='Prediction', line=dict(color='green', dash='dash')))
        fig.update_layout(title="Exchange Rate Forecast", xaxis_title="Date", yaxis_title="Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Predicted Values")
        st.dataframe(prediction_df)

        st.subheader("📉 Model Performance (Backtesting on Last 7 Days)")
        backtest_result = backtest_models(data)
        st.write(pd.DataFrame(backtest_result).T)

        symbol_key = f"{from_currency}{to_currency}"
        tradingview_symbol = TRADINGVIEW_SYMBOLS.get(symbol_key, f"FX_IDC:{symbol_key}")
        st.markdown(f"""
        <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol_key}&symbol={tradingview_symbol}&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1&hidevolume=1&showpopupbutton=1&popup_width=1000&popup_height=650"
        width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no" allowfullscreen="">
        </iframe>
        """, unsafe_allow_html=True)

        st.success("✅ Forecasting complete and performance analysis done!")
