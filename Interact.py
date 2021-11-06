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


share_amount= st.number_input('How many shares do you want?', min_value=100)   
st.header('You selected {} shares.'.format(share_amount))

def amount(share_amount):
    value = close_price(dropdown_stocks) * share_amount
    price = value
    return round(value,2)
st.text('Your initial investment is ${}'.format(amount(share_amount)))
 
dropdown_option = st.selectbox('Where do you want to reinvest your dividends?', options)


####### I believe all code should be inserted here using the if, elif , else method



#@st.cache
#if options == Same Stock
#    def stock(tickers):

dropdown_crypto = st.selectbox('What crypto would you like to reinvest in?', crypto)

        
        
dropdown_stocks_dividend = st.selectbox('Pick your stock dividend', tickers) 

#@st.cache
#def dividend(df2):
#    rel = df2.pct_change()
#    cumret = (1 + rel).cumprod() - 1
#    cumret = cumret.fillna(0)
#    return cumret
#if len(dropdown_stocks) > 0:
#    df2 = yf.download(dropdown_stocks, start, end)['dividendRate']
#    df2 = dividend(yf.download(dropdown_stocks, start, end)['dividendYield'])
#    st.header('Returns of {}'.format(dropdown_stocks))
#    st.bar_chart(df2)

#def sip(investment, tenure, interest, amount=0, is_year=True, is_percent=True, show_amount_list=False):
#    tenure = tenure*12 if is_year else tenure
#    interest = interest/100 if is_percent else interest
#    interest /= 12
#    amount_every_month = {}
#    for month in range(tenure):
#        amount = (amount + investment)*(1+interest)
#        amount_every_month[month+1] = amount
#    return {'Amount after invexting your dividends back into the chosen stock': amount, 'Amount every month': amount_every_month} #if show_amount_list else {'Amount after investing your dividends back into the chosen stock': amount} 
#years = input()
# (monthly amount, years, percent returned)

#SIP_maturity = sip(monthly_f_div, 20, average_annual_return["F"])

#print(SIP_maturity)




dropdown_div = st.multiselect('This stock will return', tickers)






#returns = last_price.pct_change().dropna()
# number of simulations
#number_simulations = 1000
#number_days = 1400

#simulation_df = pd.DataFrame()    
#for x in range(number_simulations):
#    count = 0
#    daily_volatility= returns.std()
    
#    price_series = []
    
#    for y in range(number_days):
#        if count == 1400:
#            break
#            price = price_series[count] * (1 + np.random.normal(0, daily_volatility))
#            price_series.append(price)
#            count += 1
            
#        simulation_df[x]= price_series
        
#simulation_df.head()
    
    #def relativeret(df):
#    rel = df.pct_change()
#    cumret = (1 + rel).cumprod() - 1
#    cumret = cumret.fillna(0)
#    return cumret
#if len(dropdown_stocks) > 0:
#    df = relativeret(yf.download(dropdown_stocks, start, end)['Adj Close'])
#    st.header('Cumulative returns of {}'.format(dropdown_stocks))
#    st.line_chart(df)
#if len(dropdown_stocks) > 0:
#    df3 = yf.download(dropdown_stocks, start, end)['dividendYield']['dividendRate']
#    st.bar_chart(df3)
    
#share_amount= st.slider('How many shares do you want?', min_value=10, max_value=500, value=20, step=5)