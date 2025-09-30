import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
from datetime import date

st.set_page_config(page_title="📊 Stock Analyzer", layout="wide")

st.title("📈 Stock Market Analyzer")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Data Fetching with error handling
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise ValueError("No data found. Try a different ticker or date range.")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

df = fetch_data(ticker, start_date, end_date)

if df is not None:

    # Technical Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)

    rsi = ta.momentum.RSIIndicator(df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    st.subheader(f"Showing data for {ticker}")
    st.dataframe(df.tail())

    # Candlestick Chart with Moving Averages and Bollinger Bands
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Candlestick'
    ))

    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20'))

    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], mode='lines', name='BB High', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], mode='lines', name='BB Low', line=dict(dash='dot')))

    fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # RSI Chart
    st.subheader("RSI Indicator")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(yaxis_title="RSI", xaxis_title="Date")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD Chart
    st.subheader("MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')))
    fig_macd.update_layout(yaxis_title="MACD", xaxis_title="Date")
    st.plotly_chart(fig_macd, use_container_width=True)

    # Volume Chart
    st.subheader("Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'))
    fig_vol.update_layout(yaxis_title="Volume", xaxis_title="Date")
    st.plotly_chart(fig_vol, use_container_width=True)

else:
    st.warning("No data to display.")
