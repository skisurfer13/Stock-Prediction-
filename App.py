import subprocess

# Install required modules
subprocess.run(["pip", "install", "yfinance"])
subprocess.run(["pip", "install", "prophet"])
subprocess.run(["pip", "install", "plotly"])

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from PIL import Image
import pandas as pd

image = Image.open('stock.jpeg')

st.image(image, use_column_width=True)

st.markdown('''
# Fintech Stock Price App 
This app shows the closing financial stock price values for S and P 500 companies along with the timeline.  
- These are 500 of the largest companies listed on stock exchanges in the US.
- Dataset resource: Yahoo Finance
- Added feature: Time series forecasting with fbprophet that can predict the stock price values over 15 years.
- Note: User inputs for the company to be analyzed are taken from the sidebar. It is located at the top left of the page (arrow symbol). Inputs for other features of data analysis can also be provided from the sidebar itself. 
''')
st.write('---')


@st.cache_data
def load_data():
    components = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return components.set_index("Symbol")


@st.cache_data
def load_quotes(asset):
    return yf.download(asset)


def main():
    components = load_data()

    st.sidebar.title("Options")

    if st.sidebar.checkbox("View companies list"):
        st.dataframe(
            components[["Security", "GICS Sector", "Date first added", "Founded"]]
        )

    title = st.empty()

    def label(symbol):
        a = components.loc[symbol]
        return symbol + " - " + a.Security

    st.sidebar.subheader("Select company")
    asset = st.sidebar.selectbox(
        "Click below to select a new company",
        components.index.sort_values(),
        index=3,
        format_func=label,
    )

    title.title(components.loc[asset].Security)
    if st.sidebar.checkbox("View company info", True):
        st.table(components.loc[asset])
    data0 = load_quotes(asset)
    data = data0.copy().dropna()
    data.index.name = None

    section = st.sidebar.slider(
        "Number of days for Data Analysis of stocks",
        min_value=30,
        max_value=min([5000, data.shape[0]]),
        value=1000,
        step=10,
    )

    data2 = data[-section:]["Adj Close"].to_frame("Adj Close")

    sma = st.sidebar.checkbox("Simple Moving Average")
    if sma:
        period = st.sidebar.slider(
            "Simple Moving Average period", min_value=5, max_value=500, value=20, step=1
        )
        data[f"SMA {period}"] = data["Adj Close"].rolling(period).mean()
        data2[f"SMA {period}"] = data[f"SMA {period}"].reindex(data2.index)

    sma2 = st.sidebar.checkbox("Simple Moving Average 2")
    if sma2:
        period2 = st.sidebar.slider(
            "Simple Moving Average 2 period", min_value=5, max_value=500, value=100, step=1
        )
        data[f"SMA2 {period2}"] = data["Adj Close"].rolling(period2).mean()
        data2[f"SMA2 {period2}"] = data[f"SMA2 {period2}"].reindex(data2.index)

    st.subheader("Stock Chart")
    st.line_chart(data2)

    st.subheader("Company Statistics")
    st.table(data2.describe())

    if st.sidebar.checkbox("View Historical Company Shares"):
        st.subheader(f"{asset} historical data")
        st.write(data2)


main()


# Part 2
def pre_dict():
    st.header('Stock prediction')

    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = ('AAPL', 'GOOG', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select company for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 15)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data = load_data(selected_stock)

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


#if st.button('Stock Prediction'):
pre_dict()
