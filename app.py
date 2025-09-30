
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Market Visualizer", layout='wide')

# ------------------------------ Helpers ------------------------------
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical data for a ticker using yfinance."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df.empty:
            st.error(f"No data returned for {ticker}. Check symbol or try different timeframe.")
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_multiple(tickers, period='1y', interval='1d'):
    return yf.download(tickers, period=period, interval=interval, progress=False, group_by='ticker', threads=True)


def compute_indicators(df: pd.DataFrame, ma_windows=[20,50], bb_window=20, bb_k=2):
    d = df.copy()
    d['Adj Close'] = d['Adj Close'].astype(float)
    # Moving averages
    for w in ma_windows:
        d[f'MA_{w}'] = d['Adj Close'].rolling(window=w, min_periods=1).mean()
    # Bollinger Bands
    d['BB_MID'] = d['Adj Close'].rolling(bb_window).mean()
    d['BB_STD'] = d['Adj Close'].rolling(bb_window).std()
    d['BB_UPPER'] = d['BB_MID'] + bb_k * d['BB_STD']
    d['BB_LOWER'] = d['BB_MID'] - bb_k * d['BB_STD']
    # RSI
    delta = d['Adj Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / roll_down
    d['RSI'] = 100 - (100 / (1 + rs))
    return d


def plot_candlestick(df: pd.DataFrame, ticker: str, ma_windows=[20,50], show_bb=True, show_rsi=True, theme="plotly_white"):
    df = df.copy().dropna()
    rows = 2 if show_rsi else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.7, 0.3] if show_rsi else [1])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                 name=f'{ticker}'), row=1, col=1)
    # Adjusted close line
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(width=1)), row=1, col=1)

    # Moving averages
    for w in ma_windows:
        colname = f'MA_{w}'
        if colname in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[colname], mode='lines', name=colname, line=dict(width=1)), row=1, col=1)

    # Bollinger Bands
    if show_bb and 'BB_UPPER' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(width=1), name='BB Upper', opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(width=1), name='BB Lower', opacity=0.6), row=1, col=1)
        # fill
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], showlegend=False, mode='lines', line=dict(width=0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], showlegend=False, mode='lines', fill='tonexty', line=dict(width=0), fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # RSI subplot
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash='dash', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', row=2, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(height=(700 if rows==2 else 500), template=theme, title=f'{ticker} Price Chart')
    fig.update_yaxes(fixedrange=False)
    return fig


def compute_portfolio_value(portfolio_df: pd.DataFrame, price_map: dict):
    # portfolio_df expected columns: ticker, qty, cost
    df = portfolio_df.copy()
    df['ticker'] = df['ticker'].str.upper()
    df['market_price'] = df['ticker'].map(price_map)
    df['market_value'] = df['market_price'] * df['qty']
    df['cost_basis'] = df['cost'] * df['qty']
    df['profit_loss'] = df['market_value'] - df['cost_basis']
    return df


def get_financial_ratios(ticker):
    """Try to get common ratios from yfinance info. If missing, return NaN for those fields."""
    t = yf.Ticker(ticker)
    info = t.info
    ratios = {}
    # Safe get
    ratios['shortName'] = info.get('shortName')
    ratios['marketCap'] = info.get('marketCap')
    ratios['trailingPE'] = info.get('trailingPE')
    ratios['forwardPE'] = info.get('forwardPE')
    ratios['priceToBook'] = info.get('priceToBook')
    ratios['enterpriseToRevenue'] = info.get('enterpriseToRevenue')
    ratios['pegRatio'] = info.get('pegRatio')
    ratios['dividendYield'] = info.get('dividendYield')
    ratios['beta'] = info.get('beta')
    # fallback: compute PE from price and earnings if available
    return ratios

# ------------------------------ UI ------------------------------
st.title("📈 Stock Market Visualizer")
st.markdown("A Streamlit app to fetch, analyze, visualize and export stock market data using yfinance and Plotly.")

# Sidebar: Controls
with st.sidebar:
    st.header("Data & Chart Settings")
    ticker = st.text_input("Ticker (comma separated for multiple)", value='AAPL')
    tickers = [t.strip().upper() for t in ticker.split(',') if t.strip()]
    period = st.selectbox('Period', options=['1mo','3mo','6mo','1y','2y','5y','10y','max'], index=3)
    interval = st.selectbox('Interval', options=['1d','1wk','1mo','1h','90m','30m','15m','5m'], index=0)
    ma_windows = st.multiselect('Moving average windows', options=[5,10,20,50,100,200], default=[20,50])
    show_bb = st.checkbox('Show Bollinger Bands', value=True)
    show_rsi = st.checkbox('Show RSI subplot', value=True)
    theme = st.selectbox('Plotly theme', options=['plotly','plotly_white','ggplot2','seaborn','simple_white'], index=1)
    st.markdown('---')
    st.header('Portfolio / Files')
    portfolio_file = st.file_uploader('Upload portfolio (CSV or Excel) with columns: ticker, qty, cost', type=['csv','xlsx','xls'])
    config_file = st.file_uploader('Load config (JSON)', type=['json'])
    st.markdown('---')
    st.header('Save / Share')
    config_name = st.text_input('Configuration name', value='my_config')
    save_config_btn = st.button('Save current config')

# Load config if provided
if config_file is not None:
    try:
        cfg = json.load(config_file)
        # override UI fields where reasonable
        if 'ticker' in cfg:
            tickers = [t.strip().upper() for t in cfg['ticker'].split(',') if t.strip()]
        if 'period' in cfg:
            period = cfg['period']
        if 'interval' in cfg:
            interval = cfg['interval']
        if 'ma_windows' in cfg:
            ma_windows = cfg['ma_windows']
        st.success('Config loaded — UI updated where possible.')
    except Exception as e:
        st.error(f'Failed to load config: {e}')

# Save config
if save_config_btn:
    cfg = dict(ticker=','.join(tickers), period=period, interval=interval, ma_windows=ma_windows, show_bb=show_bb, show_rsi=show_rsi, theme=theme)
    cfg_bytes = json.dumps(cfg, indent=2).encode('utf-8')
    st.download_button(label='Download config JSON', data=cfg_bytes, file_name=f'{config_name}.json', mime='application/json')

# Main layout
col1, col2 = st.columns([3,1])

with col1:
    st.subheader('Price Charts')
    if len(tickers) == 0:
        st.info('Enter at least one ticker.')
    else:
        # Single ticker: show chart and ratio
        main_ticker = tickers[0]
        df = fetch_data(main_ticker, period=period, interval=interval)
        if not df.empty:
            df_ind = compute_indicators(df, ma_windows=ma_windows)
            fig = plot_candlestick(df_ind, main_ticker, ma_windows=ma_windows, show_bb=show_bb, show_rsi=show_rsi, theme=theme)
            st.plotly_chart(fig, use_container_width=True)

            # Export options
            st.markdown('**Export chart**')
            # HTML
            html_bytes = fig.to_html(full_html=True, include_plotlyjs='cdn').encode('utf-8')
            st.download_button('Download chart as HTML', data=html_bytes, file_name=f'{main_ticker}_chart_{period}.html', mime='text/html')
            # PNG (requires kaleido)
            try:
                png_bytes = fig.to_image(format='png', engine='kaleido')
                st.download_button('Download chart as PNG', data=png_bytes, file_name=f'{main_ticker}_chart_{period}.png', mime='image/png')
            except Exception as e:
                st.warning('PNG export requires `kaleido`. Install it if you want PNG export. Error: ' + str(e))

            # Financial ratios
            st.subheader('Financial Ratios')
            with st.spinner('Fetching financial ratios...'):
                ratios = get_financial_ratios(main_ticker)
            ratio_df = pd.DataFrame([ratios])
            st.table(ratio_df.T)

        # If multiple tickers provided, show correlation analysis
        if len(tickers) > 1:
            st.subheader('Correlation Analysis')
            with st.spinner('Fetching multiple tickers...'):
                # fetch adjusted close for tickers
                data = fetch_multiple(tickers, period=period, interval=interval)
            # yfinance returns multiindex if group_by='ticker' for multiple tickers depending on call; safer to download per ticker
            adj = pd.DataFrame()
            for t in tickers:
                tmp = fetch_data(t, period=period, interval=interval)
                if not tmp.empty:
                    adj[t] = tmp['Adj Close']
            if not adj.empty:
                corr = adj.pct_change().corr()
                # plot heatmap
                fig2 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorbar=dict(title='Correlation')))
                fig2.update_layout(title='Daily Return Correlation')
                st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader('Portfolio Snapshot')
    if portfolio_file is not None:
        try:
            if portfolio_file.type in ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','application/vnd.ms-excel'):
                p_df = pd.read_excel(portfolio_file)
            else:
                p_df = pd.read_csv(portfolio_file)
            # normalize
            p_df.columns = [c.lower().strip() for c in p_df.columns]
            # expect ticker, qty, cost
            if not set(['ticker','qty','cost']).issubset(set(p_df.columns)):
                st.error('Portfolio file must contain columns: ticker, qty, cost (case-insensitive).')
            else:
                st.write(p_df)
                # fetch last price for each ticker
                price_map = {}
                for t in p_df['ticker'].str.upper().unique():
                    try:
                        last = yf.Ticker(t).history(period='1d')
                        if not last.empty:
                            price_map[t] = last['Close'].iloc[-1]
                        else:
                            price_map[t] = np.nan
                    except Exception:
                        price_map[t] = np.nan
                port_df = compute_portfolio_value(p_df, price_map)
                st.table(port_df)
                st.metric('Total Market Value', f"{port_df['market_value'].sum():,.2f}")
                st.metric('Total Profit/Loss', f"{port_df['profit_loss'].sum():,.2f}")
        except Exception as e:
            st.error(f'Failed to parse portfolio file: {e}')
    else:
        st.info('Upload a portfolio file (CSV/XLSX). Example columns: ticker, qty, cost')

# Footer: Quick correlation and batch download
st.markdown('---')
st.subheader('Batch tools')
colA, colB = st.columns(2)
with colA:
    batch_tickers = st.text_area('Batch tickers (one per line) for quick snapshot', value='AAPL\nMSFT\nGOOG')
    if st.button('Get snapshots'):
        lines = [l.strip().upper() for l in batch_tickers.splitlines() if l.strip()]
        snap = []
        for t in lines:
            try:
                info = yf.Ticker(t).info
                price = yf.Ticker(t).history(period='1d')['Close'].iloc[-1]
                snap.append({'ticker':t,'price':price,'marketCap':info.get('marketCap')})
            except Exception as e:
                snap.append({'ticker':t,'price':None,'marketCap':None,'error':str(e)})
        st.table(pd.DataFrame(snap))

with colB:
    if st.button('Download historical CSV for tickers'):
        lines = [l.strip().upper() for l in batch_tickers.splitlines() if l.strip()]
        if lines:
            combined = fetch_multiple(lines, period=period, interval=interval)
            buf = BytesIO()
            # save to CSV with multiindex flattening if necessary
            try:
                combined.to_csv(buf)
                buf.seek(0)
                st.download_button('Download CSV', data=buf, file_name=f'historical_{"_".join(lines)}.csv', mime='text/csv')
            except Exception as e:
                st.error('Failed to prepare CSV: ' + str(e))

st.markdown('---')
st.caption('Built with Streamlit + yfinance + Plotly. Configure timeframes, overlays and export charts for sharing.')

# ------------------------------ End ------------------------------
