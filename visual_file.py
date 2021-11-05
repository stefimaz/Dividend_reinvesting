import streamlit as st
import pandas as pd
import yfinance as yf


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title('Dividends Reinvestment Dashboard')
    st.text('Analysis of the power of Dividend Reinvestment.')
    st.text('This project will let you select one stock that pays dividends.')
    st.text('Once chosen, will be able to choose a few different options.')

with dataset:
    
    tickers = ("AAPL", "MSFT","MO", "F","T","XOM","LUMN","JPM")
    crypto = ("BTC-USD", "ETH-USD")
    options = ("Same Stock", "Crypto", "Keep the cash")

    dropdown_stocks = st.multiselect('Pick your stock', tickers)
    
    start = st.date_input('Start Date', value = pd.to_datetime('2011-01-01'))
    end = st.date_input('End Date', value = pd.to_datetime('today'))
  
    def relativeret(df):
        rel = df.pct_change()
        cumret = (1 + rel).cumprod() - 1
        cumret = cumret.fillna(0)
        return cumret
    if len(dropdown_stocks) > 0:
        df = relativeret(yf.download(dropdown_stocks, start, end)['Adj Close'])
        st.header('Returns of{}'.format(dropdown_stocks))
        st.line_chart(df)
        
    share_number= st.slider('How many shares do you want?', min_value=10, max_value=500, value=20, step=5)
    
    dropdown_option = st.multiselect('Where do you want to reinvest your dividends?', options)
    dropdown_crypto = st.multiselect('What crypto would you like to reinvest in?', crypto)
    dropdown_div = st.multiselect('This stock will return', tickers)

    
