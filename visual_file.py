import streamlit as st
import pandas as pd
import yfinance as yf

import os
from pathlib import Path
import requests
import json
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import hvplot.pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics


st.title('Dividends Reinvestment Dashboard')
st.text('Analysis of the power of Dividend Reinvestment.')
st.text('This project will let you select one stock that pays dividends.')
st.text('Once chosen, you will be able to select a few different options.')


    
tickers = ("AAPL", "MSFT","MO", "F","T","XOM","LUMN","JPM")
crypto = ("BTC-USD", "ETH-USD")
options = ("Same Stock", "Crypto", "Keep the cash")

dropdown_stocks = st.selectbox('Pick your stock', tickers)
     
start = st.date_input('Start Date', value= pd.to_datetime('2011-01-01'))
end = st.date_input('End Date', value= pd.to_datetime('today'))

@st.cache
def close_price(dropdown_stocks):
    data = yf.download(dropdown_stocks, period = "today", interval= "1d")['Adj Close'][0]
    price = data    
    return round(data,2)

    
if len(dropdown_stocks) > 0:
    df = yf.download(dropdown_stocks, start, end)['Adj Close']
    st.header('Historical value of {}'.format(dropdown_stocks))
    st.text('The current value is ${}'.format(close_price(dropdown_stocks)))
    st.line_chart(df)

share_amount= st.number_input('How many shares do you want?', min_value=1)   
st.header('You selected {} shares.'.format(share_amount))

def amount(share_amount):
    value = close_price(dropdown_stocks) * share_amount
    price = value
    return round(value,2)
st.text('Your total buyin will be {}'.format(amount(share_amount)))
 
dropdown_option = st.selectbox('Where do you want to reinvest your dividends?', options)
# Create and empty DataFrame for closing prices of chosen stock
df_stock_prices = pd.DataFrame()

# Fetch the closing prices for all the stocks
df_stock_prices[dropdown_option] = close_price(dropdown_stocks)

if dropdown_option == "Same Stock":
    def relativeret(df):
        rel = df.pct_change()
        cumret = (1 + rel).cumprod() - 1
        cumret = cumret.fillna(0)
        return cumret
    if len(dropdown_stocks) > 0:
        df = relativeret(yf.download(dropdown_stocks, start, end)['Adj Close'])
        st.header('Cumulative returns of {}'.format(dropdown_stocks))
        st.line_chart(df)
    st.slider('How many years of investment projections?', min_value= 10, max_value= 30, value=10, step= 10)  
    
    # simulation of return of the stock with dividends added
    
elif dropdown_option == "Crypto":
    dropdown_crypto = st.selectbox('What crypto would you like to reinvest in?', crypto)
    st.slider('How many years of investment projections?', min_value= 10, max_value= 30, value=10, step= 10)
    
    # simulation of chosen crypto using invested dividends
     
elif dropdown_option == "Keep the cash":
    dropdown_div = st.multiselect('This stock will return', tickers)
    st.slider('How many years of pocketing the cash?', min_value= 10, max_value= 30, value=10, step= 10)
    
    # simulation of dividend investment over time
    

