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

import hvplot.pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
#from MCForecastTools_2Mod import MCSimulation

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
    st.text(f'The average yearly dividend for {dropdown_stocks} is:')
 
    tickerData = yf.Ticker(dropdown_stocks) # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start, end=end) #get the historical prices for this ticker
    
    # Calculate the yearly % after getting the value from yahoo finance
    string_summary = tickerData.info['dividendYield']
    yearly_div = (string_summary * 4) * 100
    st.info(f'{yearly_div}%')
    
# Asking the user for desired amount of share to purchase, showing 100 shares to start. minimum will be 10 shares    
share_amount= st.number_input('How many shares do you want?',value=100, min_value=10)   
st.header('You selected {} shares.'.format(share_amount)) 

@st.cache
# Calculating the value of the investment compare to the amount of share selected, giving the amount
def amount(share_amount):
    value = close_price(dropdown_stocks) * share_amount
    price = value
    return round(value,2)

initial_investment = (amount(share_amount))
st.info(f'Your initial investment is ${initial_investment}')

# Showing amount of yearly dividend in $  
st.text(f'Your current yearly dividend for the amount of shares you selected is:')
 
# Calculate the yearly $ after getting the value from yahoo finance    
string_summary2 = tickerData.info['dividendRate']
yearly_div_amount = (string_summary2) * (share_amount)
st.info(f'${yearly_div_amount}')        

# This is where the user make the choice of where to reinvest the dividend paid. 
dropdown_option = st.selectbox('Where do you want to reinvest your dividends?', options)

# Create and empty DataFrame for closing prices of chosen stock
df_stock_prices = pd.DataFrame()

# Fetch the closing prices for all the stocks
df_stock_prices[dropdown_option] = close_price(dropdown_stocks)


# Calculating the cumulative returns after choosing the same stock option
if dropdown_option == "Same Stock":
    @st.cache
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
    daily = yf.download(dropdown_stocks, start, end)['Adj Close']
    def average_annual (daily):
        rel = daily.pct_change()
        ave_rel= rel.mean()
        anual_ret = (ave_rel * 252) * 100
        return anual_ret
    yearly_returns = average_annual(daily)
    st.subheader(f'Average yearly returns of {dropdown_stocks} is {average_annual(daily): .2f}%')
    
    # Slider 1 with option to select the amount of year to reinvest(10, 20 or 30)
    year_opt1 = st.slider('How many years of investment projections?', min_value= 10, max_value= 30, value=10, step= 10) 

    
    # simulation of return of the stock with dividends to be added here 
    same_amount_div = yearly_div_amount / 12
    same_interest = yearly_returns
    investment3 = initial_investment
    @st.cache
    def same_stock(investment, tenure, interest, amount=investment3, is_year=True, is_percent=True, show_amount_list=False):
        tenure = tenure*12 if is_year else tenure
        interest = interest/100 if is_percent else interest
        interest /= 12
        amount_every_month = {}
        for month in range(tenure):
            amount = (amount + investment)*(1+interest)
            amount_every_month[month+1] = amount
        return {f'A': amount,
                'Amount every month': amount_every_month} if show_amount_list else round(amount, 2) 
    # (monthly amount, years, percent returned)
    Same_maturity = same_stock(same_amount_div, year_opt1, same_interest)
    
    st.subheader(f'Your stock projection for {year_opt1} after reinvesting the dividends will be:$ {Same_maturity}')
    
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
    
    
    # Calculate the annual average return data for the stocks
    # Use 252 as the number of trading days in the year    
    daily = yf.download(dropdown_crypto, start, end)['Adj Close']
    def average_annual_crypto (daily):
        rel = daily.pct_change()
        ave_rel= rel.mean()
        anual_ret = (ave_rel * 252) * 100
        return anual_ret
    crypto_yearly_returns = average_annual_crypto(daily)
    st.subheader(f'Average yearly returns of {dropdown_crypto} is {average_annual_crypto(daily): .2f}%')   
   
    investment = yearly_div_amount / 12
    interestcrypto = crypto_yearly_returns
    # simulation of dividend investment over time. 
    # simple dividend reinvestment function
    @st.cache
    def sip_crypto(investment, tenure, interest, amount=0, is_year=True, is_percent=True, show_amount_list=False):
        tenure = tenure*12 if is_year else tenure
        interest = interest/100 if is_percent else interest
        interest /= 12
        amount_every_month = {}
        for month in range(tenure):
            amount = (amount + investment)*(1+interest)
            amount_every_month[month+1] = amount
        return {f'A': amount,
                'Amount every month': amount_every_month} if show_amount_list else round(amount, 2) 
    # (monthly amount, years, percent returned)
    SIP_crypto = sip_crypto(investment, year_opt2, interestcrypto)
    
    
    st.header(f'The projected return for dividends reinvested into {dropdown_crypto} is {SIP_crypto}')

    
    
    
    
# Calculating the projected return for reinvestment into the same stock chosen here
elif dropdown_option == "Keep the cash":
    
    # Slider 3 with option to select the amount of year to reinvest(10, 20 or 30)
    year_opt3 = st.slider('How many years of pocketing the cash?', min_value= 10, max_value= 30, value=10, step= 10)
    st.write(f'You will reinvest your dividends for {year_opt3} years')
    
    
    daily = yf.download(dropdown_stocks, start, end)['Adj Close']
    def average_annual (daily):
        rel = daily.pct_change()
        ave_rel= rel.mean()
        anual_ret = (ave_rel * 252) * 100
        return anual_ret
    
    yearly_returns = average_annual(daily)
    investment1 = initial_investment
    interest1 =  yearly_returns
    
    def sip_stock(investment, tenure, interest, amount= investment1, is_year=True, is_percent=True, show_amount_list=False):
        tenure = tenure*12 if is_year else tenure
        interest = interest/100 if is_percent else interest
        interest /= 12
        amount_every_month = {}
        for month in range(tenure):
            amount = (amount + investment)*(1+interest)
            amount_every_month[month+1] = amount
        return {f'A': amount,
                'Amount every month': amount_every_month} if show_amount_list else round(amount, 2) 
    # (monthly amount, years, percent returned)
    SIP_stock_maturity = sip_stock(0, year_opt3, interest1)
    
    st.header(f'The projected return for {dropdown_stocks} is {SIP_stock_maturity}')
#    st.subheader(f'Your total dividend return will be {SIP_maturity}')

         
    investment = yearly_div_amount / 12
    interest = 0
    # simulation of dividend investment over time. 
    # simple dividend reinvestment function
    @st.cache
    def sip(investment, tenure, interest, amount=0, is_year=True, is_percent=True, show_amount_list=False):
        tenure = tenure*12 if is_year else tenure
        interest = interest/100 if is_percent else interest
        interest /= 12
        amount_every_month = {}
        for month in range(tenure):
            amount = (amount + investment)*(1+interest)
            amount_every_month[month+1] = amount
        return {f'A': amount,
                'Amount every month': amount_every_month} if show_amount_list else round(amount, 2) 
    # (monthly amount, years, percent returned)
    SIP_maturity = sip(investment, year_opt3, interest)
    
    st.subheader(f'Your total dividend return will be {SIP_maturity}')
    
# iinclude a summary of the 3 options
   # we should have a projection of the stock over the next chosen period as well to show the user wher they will be
