import streamlit as st
import pandas as pd
import yfinance as yf
import sqlalchemy
import cufflinks as cf
import datetime
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import hvplot.pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

# title of the project and introduction on what to do 

st.title('Dividends Reinvestment Dashboard')
st.write('Analysis of the **Power** of **Dividend Reinvestment**.')
st.write('Select from the list of stocks that pays dividends.')
st.write('You will then be able to select between three options.')
st.write('***Choose wisely***.')

# chosen stock and crypto tickers. choice of the 3 different options

tickers = ("AAPL","F","JPM","LUMN","MO","MSFT","T")
crypto = ("BTC-USD", "ETH-USD")
options = ("Same Stock", "Crypto", "Keep the cash")

# box selection for the stock to invest in
dropdown_stocks = st.selectbox('Pick your stock', tickers)

# starting date of the stock history. This is interactive and can be changed by the user
start = st.date_input('Start Date', value= pd.to_datetime('2011-01-01'))
end = st.date_input('End Date', value= pd.to_datetime('today'))

# option to have a fix time period for historical data
# start= pd.to_datetime('2011-01-01')
# end= pd.to_datetime('today')

# this is a cache so the page does not reload the entire data if it is not changed
@st.cache

# function to let the user choose the stock
def close_price(dropdown_stocks):
    data = yf.download(dropdown_stocks, period = "today", interval= "1d")['Adj Close'][0]
    price = data    
    return round(data,2)

# this will display the chosen stock, the value of the stock, and a line chart of the price history    
if len(dropdown_stocks) > 0:
    df = yf.download(dropdown_stocks, start, end)['Adj Close']
    st.header('Historical value of {}'.format(dropdown_stocks))
    st.info('The current value is ${}'.format(close_price(dropdown_stocks)))
    st.line_chart(df)
    
    # Showing what is the yearly dividend % for the chosen stock
    st.text('The average yearly dividend in % for that stock is:')
 
    tickerData = yf.Ticker(dropdown_stocks) # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start, end=end) #get the historical prices for this ticker
    
    # Calculate the yearly % after getting the value from yahoo finance
    string_summary = tickerData.info['dividendYield']
    yearly_div = (string_summary * 4) * 100
    st.info(yearly_div)
    
# Asking the user for desired amount of share to purchase, showing 100 shares to start. minimum will be 10 shares    
share_amount= st.number_input('How many shares do you want?',value=100, min_value=10)   
st.header('You selected {} shares.'.format(share_amount)) 

# Calculating the value of the investment compare to the amount of share selected, giving the amount
def amount(share_amount):
    value = close_price(dropdown_stocks) * share_amount
    price = value
    return round(value,2)
st.info('Your initial investment is ${}'.format(amount(share_amount)))

# Showing amount of yearly dividend in $  
st.text('Your current yearly dividend for the amount of shares you selected is:')
 
# Calculate the yearly $ after getting the value from yahoo finance    
string_summary2 = tickerData.info['dividendRate']
yearly_div_amount = (string_summary2 * 4) * (share_amount)
st.info(yearly_div_amount)        


# This is where the user make the choice of where to reinvest the dividend paid. 
dropdown_option = st.selectbox('Where do you want to reinvest your dividends?', options)

# Create and empty DataFrame for closing prices of chosen stock
df_stock_prices = pd.DataFrame()

# Fetch the closing prices for all the stocks
df_stock_prices[dropdown_option] = close_price(dropdown_stocks)

# Calculating the cumulative returns after choosing the same stock option
if dropdown_option == "Same Stock":
    def relativeret(df):
        rel = df.pct_change()
        cumret = (1 + rel).cumprod() - 1
        cumret = cumret.fillna(0)
        return cumret
    
    # Showing the plot of the cumulative returns
    if len(dropdown_stocks) > 0:
        df = relativeret(yf.download(dropdown_stocks, start, end)['Adj Close'])
        st.header('Cumulative returns of {}'.format(dropdown_stocks))
        st.line_chart(df)
     
    # Calculate the annual average return data for the stocks
    # Use 252 as the number of trading days in the year    
    # Still working on this one but feel free to make it work  :) 
    def average_annual ():
        rel = df2.pcs_change()
        ave_rel= rel.mean()
        anual_ret = (ave_rel * 252) * 100
        return anual_ret
    st.header('Average yearly returns of {}'.format(average_annual(df2)))
    
    # Slider 1 with option to select the amount of year to reinvest(10, 20 or 30)
    year_opt1 = st.slider('How many years of investment projections?', min_value= 10, max_value= 30, value=10, step= 10) 

    
    # simulation of return of the stock with dividends to be added here 
 
    # Calculating the projected return for crypto opyion chosen here
elif dropdown_option == "Crypto":
    
    # selection of the crypto to reinvest in
    dropdown_crypto = st.selectbox('What crypto would you like to reinvest in?', crypto)
    
    # Getting the data for selected crypto from yahoo finance and ploting it as a line chart
    if len(dropdown_crypto) > 0:
        df = yf.download(dropdown_crypto, start, end)['Adj Close']
        st.header('Historical value of {}'.format(dropdown_crypto))
        st.line_chart(df)
        
        # Slider 2 with option to select the amount of year to reinvest(10, 20 or 30)
    year_opt2 = st.slider('How many years of investment projections?', min_value= 10, max_value= 30, value=10, step= 10)
    
    
    # simulation of chosen crypto using invested dividends to be added here
     
# Calculating the projected return for reinvestment into the same stock chosen here
elif dropdown_option == "Keep the cash":
    
    # Slider 3 with option to select the amount of year to reinvest(10, 20 or 30)
    year_opt3 = st.slider('How many years of pocketing the cash?', min_value= 10, max_value= 30, value=10, step= 10)
    st.header('This stock will return', tickers)
    
    # simulation of dividend investment over time. 
    # simple dividend reinvestment function
    # gives error for now because some names need to be changed
    def sip(investment, tenure, interest, amount=0, is_year=True, is_percent=True, show_amount_list=False):
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
   
    # we should have a projection of the stock over the next chosen period (10, 20 30) as well to show the user wher they will be
     

