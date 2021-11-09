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

#i commented out line 95-96 in the MCForecast file to avoid printing out lines "Running simulation number"

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
    st.text(f'The average yearly dividend {dropdown_stocks} is:')
 
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

st.info('Your initial investment is ${}'.format(amount(share_amount)))

# Showing amount of yearly dividend in $  
st.text('Your current yearly dividend for the amount of shares you selected is:')
 
# Calculate the yearly $ after getting the value from yahoo finance    
string_summary2 = tickerData.info['dividendRate']
yearly_div_amount = (string_summary2 * 4) * (share_amount)



#Predict stock using series of Monte Carlo simulation. Only works with one stock at a time.
def mc_stock_price(years, simulations):
#     historic_end = pd.to_datetime("today")
#     historic_start = historic_end - np.timedelta64(4,"Y")
    for i in dropdown_stocks:
    #calling historic data

    stock_hist =  stock.history(start = historic_start, end = historic_end)
    
    #data-cleaning
    stock_hist.drop(columns = ["Dividends","Stock Splits"], inplace = True)
    stock_hist.rename(columns = {"Close":"close"}, inplace = True)
    stock_hist = pd.concat({i: stock_hist}, axis = 1)
    
    #defining variables ahead of time in preparation for MC Simulation series
    Upper_Yields = []
    Lower_Yields = []
    Means = []
    currentYear = datetime.datetime.now().year
    Years = [currentYear]
    
    #beginning Simulation series and populating with outputs
    
    #for x in range(number of years)
    for x in range(years):
        MC_looped = MCSimulation(portfolio_data = stock_hist, 
                                      num_simulation= simulations,
                                      num_trading_days= 252*x+1)
        MC_summary_stats = MC_looped.summarize_cumulative_return()
        Upper_Yields.append(MC_summary_stats["95% CI Upper"])
        Lower_Yields.append(MC_summary_stats["95% CI Lower"])
        Means.append(MC_summary_stats["mean"])
        Years.append(currentYear+(x+1))
    
    potential_upper_price = [element * stock_hist[i]["close"][-1] for element in Upper_Yields]
    potential_lower_price = [element * stock_hist[i]["close"][-1] for element in Lower_Yields]
    potential_mean_price = [element * stock_hist[i]["close"][-1] for element in Means]
    
    print(i, potential_lower_price)
    
    plt.figure(figsize= (20,10))
    plt.title(i + " Forecast Price")
    plt.xlabel = "Years Forecasted"
    plt.ylabel = "Price"
    plt.xticks(ticks = list(range(4)), labels = Years)
    plt.plot(potential_lower_price, linestyle = ":", color = "r", label = "95% Lower CI")
    plt.plot(potential_upper_price, linestyle = ":", color = "b", label = "95% Upper CI")
    plt.plot(potential_mean_price, color = "g", label = "Mean")
    plt.legend(loc="upper left")
    plt.show()
    
        
    


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
    daily = yf.download(dropdown_stocks, start, end)['Adj Close']
    def average_annual (daily):
        rel = daily.pct_change()
        ave_rel= rel.mean()
        anual_ret = (ave_rel * 252) * 100
        return anual_ret
    st.subheader(f'Average yearly returns of {dropdown_stocks} is {average_annual(daily): .2f}%')
    
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
    st.write(f'You will reinvest your dividends for {year_opt3} years')
    
    st.header(f'The projected return for {dropdown_stocks} is', tickers)
    
    investment = yearly_div_amount / 12
    interest = 5
    
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
   
    # we should have a projection of the stock over the next chosen period as well to show the user wher they will be
     

