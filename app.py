# stock_visualizer_extended.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

# ---------------- Helper Functions ----------------
def sma(series, window): return series.rolling(window).mean()
def ema(series, window): return series.ewm(span=window, adjust=False).mean()

def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma, ma + n_std*std, ma - n_std*std

def rsi(series, window=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up, ma_down = up.ewm(alpha=1/window, adjust=False).mean(), down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100/(1+rs))

# ---------------- Charts ----------------
def plot_candlestick(df, ticker, sma_list=None, ema_list=None, boll=False):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name=ticker
    ))
    if sma_list:
        for w in sma_list:
            fig.add_trace(go.Scatter(x=df.index, y=sma(df['Close'], w), mode='lines', name=f"SMA {w}"))
    if ema_list:
        for w in ema_list:
            fig.add_trace(go.Scatter(x=df.index, y=ema(df['Close'], w), mode='lines', name=f"EMA {w}"))
    if boll:
        ma, upper, lower = bollinger_bands(df['Close'])
        fig.add_trace(go.Scatter(x=df.index, y=upper, line=dict(dash="dot"), name="BB Upper"))
        fig.add_trace(go.Scatter(x=df.index, y=lower, line=dict(dash="dot"), name="BB Lower"))
    fig.update_layout(title=f"{ticker} Candlestick", xaxis_rangeslider_visible=False)
    return fig

def plot_rsi(df):
    r = rsi(df['Close'])
    fig = go.Figure([go.Scatter(x=r.index, y=r, name="RSI")])
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    return fig

# ---------------- Streamlit App ----------------
st.title("📊 Stock Market Visualizer - Advanced")

if "data" not in st.session_state:
    st.warning("⚠️ First run stock_data_app.py to fetch and save data into session_state.")
else:
    df = st.session_state["data"]
    ticker = st.text_input("Ticker for Chart", "AAPL")

    # Candlestick
    sma_list = st.multiselect("SMA", [10,20,50])
    ema_list = st.multiselect("EMA", [10,20,50])
    boll = st.checkbox("Show Bollinger Bands")
    st.plotly_chart(plot_candlestick(df, ticker, sma_list, ema_list, boll))

    # RSI
    if st.checkbox("Show RSI"):
        st.plotly_chart(plot_rsi(df))

    # Portfolio Tracking
    st.header("📂 Portfolio Tracker")
    file = st.file_uploader("Upload Portfolio CSV/Excel", type=["csv","xlsx"])
    if file:
        portfolio = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.write(portfolio)

    # Ratios
    st.header("📑 Financial Ratios")
    if st.button("Get Ratios"):
        stock = yf.Ticker(ticker)
        info = stock.info
        st.write({
            "Market Cap": info.get("marketCap"),
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
        })

    # Correlation
    st.header("🔗 Correlation Analysis")
    tickers = st.text_input("Enter tickers (comma separated)", "AAPL,MSFT,GOOG").split(",")
    if st.button("Show Correlation"):
        prices = pd.DataFrame()
        for t in tickers:
            prices[t.strip()] = yf.download(t.strip(), period="1y")["Close"]
        corr = prices.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

    # Export
    st.header("📥 Export Charts")
    chart = plot_candlestick(df, ticker)
    png = chart.to_image(format="png")
    html = chart.to_html()
    st.download_button("Download PNG", data=png, file_name="chart.png")
    st.download_button("Download HTML", data=html, file_name="chart.html")

    # Config Save/Load
    st.header("⚙️ Configurations")
    config = {"sma": sma_list, "ema": ema_list, "bollinger": boll}
    if st.button("Save Config"):
        with open("config.json","w") as f: json.dump(config,f)
        st.success("Config saved")
    if st.button("Load Config"):
        try:
            config = json.load(open("config.json"))
            st.write(config)
        except:
            st.error("No config found")
