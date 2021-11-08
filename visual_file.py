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
st.write('Analysis of the **Power** of **Dividend Reinvestment**.')
st.write('Select from the list of stocks that pays dividends.')
st.write('You will then be able to select between three options.')
st.write('***Choose wisely***.')

tickers = ("AAPL","F","JPM","LUMN","MO","MSFT","T")
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
    st.text('Dividends paid during the most recent four quarters...')
    
    def ticker_fn(Amount: float):
        return Amount

    col_list = ['Amount']

    ticker_fn = "Resources/"+dropdown_stocks+"_dividends.csv"
    ticker_div=pd.read_csv(ticker_fn, usecols=col_list)
    #ticker_fn.loc['Amount'] = ticker_fn.loc['Amount'].astype('float')

    st.text(ticker_div.head(4))
    
    
    ticker_div['Amount'].head(1)
    latest_div = ticker_div['Amount'].head(1)


    
share_amount= st.number_input('How many shares do you want?', min_value=100)   
st.header('You selected {} shares.'.format(share_amount))

def amount(share_amount):
    value = close_price(dropdown_stocks) * share_amount
    price = value
    return round(value,2)
st.text('Your total investment will be {}'.format(amount(share_amount)))
st.text(latest_div)
st.text('You will receive {} in dividends on a quarterly basis'.format(latest_div))

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
    
    if len(dropdown_crypto) > 0:
        df = yf.download(dropdown_crypto, start, end)['Adj Close']
        st.header('Historical value of {}'.format(dropdown_crypto))
        st.line_chart(df)
    st.slider('How many years of investment projections?', min_value= 10, max_value= 30, value=10, step= 10)
    
    
    # simulation of chosen crypto using invested dividends
     
elif dropdown_option == "Keep the cash":
    st.slider('How many years of pocketing the cash?', min_value= 10, max_value= 30, value=10, step= 10)
    st.header('This stock will return', tickers)
    
    
    
    def dip(investment, tenure, interest, amount=0, is_year=True, is_percent=True, show_amount_list=False):
        tenure = tenure*12 if is_year else tenure
        interest = interest/100 if is_percent else interest
        interest /= 12
        amount_every_month = {}
        for month in range(tenure):
            amount = (amount + investment)*(1+interest)
            amount_every_month[month+1] = amount
        return {'Amount after invexting your dividends back into the chosen stock': amount, 'Amount every month': amount_every_month} 
    SIP_maturity = sip(5, 10, 5)
    st.header('Your total return will be {}'.format(sip(investment, tenure, interest)))
    # simulation of dividend investment over time
    

