import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import json
import io

# ----------------------
# Sidebar – App Settings
# ----------------------
st.set_page_config(page_title="Stock Market Visualizer", layout="wide")
st.title("📈 Stock Market Visualizer")

with st.sidebar:
    st.header("Stock Selection & Settings")
    symbol = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL").upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    show_ma = st.checkbox("Show Moving Averages", value=True)
    show_rsi = st.checkbox("Show RSI", value=False)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=False)

    st.markdown("---")
    st.subheader("Chart Customization")
    ma1 = st.slider("MA Short Window", 5, 50, 20)
    ma2 = st.slider("MA Long Window", 10, 200, 50)

    st.markdown("---")
    st.subheader("Export & Save")
    export_format = st.selectbox("Export chart as", ["None", "PNG", "HTML"])
    if st.button("Save Configuration"):
        config = {
            "symbol": symbol,
            "start": str(start_date),
            "end": str(end_date),
            "ma1": ma1,
            "ma2": ma2,
            "show_ma": show_ma,
            "show_rsi": show_rsi,
            "show_bollinger": show_bollinger
        }
        config_str = json.dumps(config)
        st.download_button("Download Config JSON", config_str, file_name="stock_config.json")

# ----------------------
# Load Stock Data
# ----------------------
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.dropna(inplace=True)
    return data

df = load_data(symbol, start_date, end_date)

if df.empty:
    st.error("No data found. Please check the ticker or date range.")
    st.stop()

# ----------------------
# Technical Indicators
# ----------------------
if show_ma:
    df[f"MA{ma1}"] = df['Close'].rolling(window=ma1).mean()
    df[f"MA{ma2}"] = df['Close'].rolling(window=ma2).mean()

if show_bollinger:
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

if show_rsi:
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

# ----------------------
# Plotly Candlestick Chart
# ----------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Candlestick"
))

if show_ma:
    fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{ma1}"], mode='lines', name=f"MA{ma1}"))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{ma2}"], mode='lines', name=f"MA{ma2}"))

if show_bollinger:
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_high'], line=dict(dash='dot'), name="Bollinger High"))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_low'], line=dict(dash='dot'), name="Bollinger Low"))

fig.update_layout(title=f"{symbol} Stock Price", xaxis_rangeslider_visible=False)

# ----------------------
# Display Chart
# ----------------------
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# RSI Chart (Optional)
# ----------------------
if show_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')))
    fig_rsi.update_layout(title="RSI Indicator", yaxis=dict(range=[0, 100]), height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

# ----------------------
# Export Chart
# ----------------------
if export_format == "PNG":
    fig.write_image("chart.png")
    with open("chart.png", "rb") as f:
        st.download_button("Download PNG", f, file_name="chart.png")
elif export_format == "HTML":
    html_bytes = fig.to_html(full_html=False).encode("utf-8")
    st.download_button("Download HTML", html_bytes, file_name="chart.html")

# ----------------------
# Portfolio Tracking
# ----------------------
st.markdown("## 📊 Portfolio Tracker")
portfolio_file = st.file_uploader("Upload Portfolio CSV (columns: Symbol, Shares)", type=["csv", "xlsx"])

if portfolio_file:
    try:
        if portfolio_file.name.endswith(".csv"):
            portfolio_df = pd.read_csv(portfolio_file)
        else:
            portfolio_df = pd.read_excel(portfolio_file)

        if not {"Symbol", "Shares"}.issubset(portfolio_df.columns):
            st.error("CSV must contain 'Symbol' and 'Shares' columns.")
        else:
            portfolio_df['Symbol'] = portfolio_df['Symbol'].str.upper()
            prices = []
            for s in portfolio_df['Symbol']:
                try:
                    data = yf.Ticker(s).history(period="1d")
                    prices.append(data['Close'][-1])
                except:
                    prices.append(0)
            portfolio_df['Price'] = prices
            portfolio_df['Value'] = portfolio_df['Price'] * portfolio_df['Shares']
            total = portfolio_df['Value'].sum()
            st.dataframe(portfolio_df)
            st.metric("Total Portfolio Value", f"${total:,.2f}")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# ----------------------
# Correlation Matrix
# ----------------------
st.markdown("## 🔗 Correlation Matrix")
symbols_input = st.text_input("Enter comma-separated tickers (e.g., AAPL,MSFT,GOOGL)", value="AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in symbols_input.split(",")]

if len(tickers) >= 2:
    correlation_data = {}
    for t in tickers:
        try:
            hist = yf.download(t, start=start_date, end=end_date)['Close']
            correlation_data[t] = hist
        except:
            continue
    if correlation_data:
        corr_df = pd.DataFrame(correlation_data).dropna().corr()
        st.dataframe(corr_df)
        st.plotly_chart(go.Figure(data=go.Heatmap(z=corr_df.values,
                                                  x=corr_df.columns,
                                                  y=corr_df.index,
                                                  colorscale="Viridis")), use_container_width=True)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly, and yFinance")
