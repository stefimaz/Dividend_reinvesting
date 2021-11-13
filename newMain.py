import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

import os
from pathlib import Path
import requests

import hvplot.pandas
import numpy as np
import matplotlib.pyplot as plt
from MCForecastTools_2Mod import MCSimulation
import plotly.express as px
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

#i commented out line 95-96 in the MCForecast file to avoid printing out lines "Running simulation number"

# title of the project and introduction on what to do 

st.title('Dividends Reinvestment Dashboard')
st.write('Analysis of the **Power** of **Dividend Reinvestment**.')
st.write('Select from the list of stocks that pays dividends.')
st.write('You will then be able to select between three options.')
st.write('***Choose wisely***.')

# chosen stock and crypto tickers. choice of the 3 different options

tickers = ("AAPL","F","JPM","LUMN","MO","MSFT","T","XOM")
crypto = ("BTC-USD", "ETH-USD", "BNB-USD")
options = ("Keep the cash", "Same Stock", "Crypto")

#dropdown_crypto = st.selectbox('',crypto)    
#crypto_forecast = predict_crypto(dropdown_crypto, year_opt2)    
#crypto_gain = round( ((float(crypto_forecast["Forecasted Values"][-1:]) - float(crypto_forecast["Forecasted Values"][0])) / crypto_forecast["Forecasted Values"][0]) * 100 , 2)    
#crypto_future = {round(yearly_div_amount*crypto_gain,2)}      
#SIP_stock_maturity = 0.0
#SIP_maturity = 0.0

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

tickerData = yf.Ticker(dropdown_stocks) # Get ticker data
stock_name = tickerData.info['shortName']
# this will display the chosen stock, the value of the stock, and a line chart of the price history    
if len(dropdown_stocks) > 0:
    df = yf.download(dropdown_stocks, start, end)['Adj Close']
    st.subheader(f'Historical value of {dropdown_stocks} ({stock_name})')
    st.info('The current value is ${}'.format(close_price(dropdown_stocks)))
    st.line_chart(df)
    
    # Showing what is the yearly dividend % for the chosen stock
    st.text(f'The average yearly yield {dropdown_stocks} is:')
 
    tickerData = yf.Ticker(dropdown_stocks) # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start, end=end) #get the historical prices for this ticker
    
    # Calculate the yearly % after getting the value from yahoo finance
    string_summary = tickerData.info['dividendYield']
    yearly_div = (string_summary) * 100
    st.info(f'{yearly_div: ,.2f}%')
    
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
st.info('Your initial investment is ${}'.format(amount(share_amount)))

# Showing amount of yearly dividend in $  
st.text(f'Your current yearly dividend for the amount of shares you selected is $:')
 
# Calculate the yearly $ after getting the value from yahoo finance    
string_summary2 = tickerData.info['dividendRate']
yearly_div_amount = (string_summary2) * (share_amount)
st.info(f'${yearly_div_amount}') 


#Predict stock using series of Monte Carlo simulation. Only works with one stock at a time.

def mc_stock_price(years):

    stock = yf.Ticker(dropdown_stocks)
    stock_hist =  stock.history(start = start, end = end)


    stock_hist.drop(columns = ["Dividends","Stock Splits"], inplace = True)
    stock_hist.rename(columns = {"Close":"close"}, inplace = True)
    stock_hist = pd.concat({dropdown_stocks: stock_hist}, axis = 1)
    
    #defining variables ahead of time in preparation for MC Simulation series
    Upper_Yields = []
    Lower_Yields = []
    Means = []
    currentYear = datetime.datetime.now().year
    Years = [currentYear]
    iteration = []
    
    for i in range(years+1):
        iteration.append(i)
        
        
    #beginning Simulation series and populating with outputs
    
    #for x in range(number of years)
    for x in range(years):
        MC_looped = MCSimulation(portfolio_data = stock_hist, 
                                        num_simulation= 100,
                                        num_trading_days= 252*x+1)
        MC_summary_stats = MC_looped.summarize_cumulative_return()
        Upper_Yields.append(MC_summary_stats["95% CI Upper"])
        Lower_Yields.append(MC_summary_stats["95% CI Lower"])
        Means.append(MC_summary_stats["mean"])
        Years.append(currentYear+(x+1))
    
    
    potential_upper_price = [element * stock_hist[dropdown_stocks]["close"][-1] for element in Upper_Yields]
    potential_lower_price = [element * stock_hist[dropdown_stocks]["close"][-1] for element in Lower_Yields]
    potential_mean_price = [element * stock_hist[dropdown_stocks]["close"][-1] for element in Means]

    prices_df = pd.DataFrame(columns = ["Lower Bound Price", "Upper Bound Price", "Forecasted Average Price"])
    prices_df["Lower Bound Price"] = potential_lower_price
    prices_df["Forecasted Average Price"] = potential_mean_price
    prices_df["Upper Bound Price"] = potential_upper_price

    fig = px.line(prices_df)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = iteration,
            ticktext = Years
        )
    )
    
    st.write(fig)
    
    
    return prices_df

def cumsum_shift(s, shift = 1, init_values = [0]):
    s_cumsum = pd.Series(np.zeros(len(s)))
    for i in range(shift):
        s_cumsum.iloc[i] = init_values[i]
    for i in range(shift,len(s)):
        s_cumsum.iloc[i] = s_cumsum.iloc[i-shift] + s.iloc[i]
    return s_cumsum


def predict_crypto(crypto, forecast_period=0):
    forecast_days = forecast_period * 365
    
    btc_data = yf.download(crypto, start, end)
    btc_data["Date"] = btc_data.index
    btc_data["Date"] = pd.to_datetime(btc_data["Date"])
    btc_data["year"] = btc_data["Date"].dt.year
    
    years = btc_data["year"].max() - btc_data["year"].min()
    
    btc_data.reset_index(inplace = True, drop = True)
    btc_data.drop(columns = ["Open", "High", "Low", "Adj Close", "Volume"], inplace = True)
    
    btc_rolling = btc_data["Close"].rolling(window = round(len(btc_data.index)/100)).mean().dropna()
    btc_change = btc_rolling.pct_change().dropna()
    btc_cum = (1 + btc_change).cumprod()
    
    list = []
    for i in range(years):
        list.append(btc_cum[i*round(len(btc_cum)/years):(i+1)*round(len(btc_cum)/years)].mean())

    slope = (list[-1]-list[0])/len(list)

    list2 = []
    for i in range(years):
        list2.append((slope*i)+list[0])
    
    upper = []
    lower = []
    for i in range(years):
        lower.append((slope*i) + (list[0]-slope))
        upper.append((slope*i) + (list[0]+slope))
        
    counter = 0
    positions = []
    for i in range(1, years):
        if (list[i] >= lower[i]) & (list[i] <= upper[i]):
            positions.append(i*round(len(btc_cum)/years))
            positions.append((i+1)*round(len(btc_cum)/years))
            counter+=1
    
    if (counter < years/2):
        btc_rolling = btc_data["Close"][positions[-2]:].rolling(window = round(len(btc_data.index)/100)).mean().dropna()
    
    if forecast_period == 0:
        
        auto_model = pm.auto_arima(btc_rolling)

        model_str = str(auto_model.summary())

        model_str = model_str[model_str.find("Model:"):model_str.find("Model:")+100]

        start_find = model_str.find("(") + len("(")
        end_find = model_str.find(")")
        substring = model_str[start_find:end_find]

        arima_order = substring.split(",")

        for i in range(len(arima_order)):
            arima_order[i] = int(arima_order[i])
    
        arima_order = tuple(arima_order)
    
        train = btc_rolling[:int(0.8*len(btc_rolling))]
        test = btc_rolling[int(0.8*len(btc_rolling)):]
        
#         test_length = 

        model = ARIMA(train.values, order=arima_order)
        model_fit = model.fit(disp=0)
    
    
    
#         if ( float(0.2*len(btc_rolling)) < int(0.2*len(btc_rolling))):
        fc, se, conf = model_fit.forecast(len(test.index), alpha=0.05)  # 95% conf
#         else:
#             fc, se, conf = model_fit.forecast((int(0.2*len(btc_rolling))), alpha=0.05)

        fc_series = pd.Series(fc, index=test.index)
        lower_series = pd.Series(conf[:, 0], index=test.index)
        upper_series = pd.Series(conf[:, 1], index=test.index)
    
        plt.rcParams.update({'font.size': 40})
        fig = plt.figure(figsize=(40,20), dpi=100)
        ax = fig.add_subplot(1,1,1)
        l1 = ax.plot(train, label = "Training")
        l2 = ax.plot(test, label = "Testing")
        l3 = ax.plot(fc_series, label = "Forecast")
        ax.fill_between(lower_series.index, upper_series, lower_series,
                         color='k', alpha=.15)
        ax.set_title('Forecast vs Actuals')
        fig.legend(loc='upper left', fontsize=40), (l1,l2,l3)
        plt.rc('grid', linestyle="-", color='black')
        plt.grid(True)
        st.write(fig)
    


    else:
        auto_model = pm.auto_arima(btc_rolling)

        model_str = str(auto_model.summary())

        model_str = model_str[model_str.find("Model:"):model_str.find("Model:")+100]

        start_find = model_str.find("(") + len("(")
        end_find = model_str.find(")")
        substring = model_str[start_find:end_find]

        arima_order = substring.split(",")

        for i in range(len(arima_order)):
            arima_order[i] = int(arima_order[i])

        arima_order = tuple(arima_order)
    
    
        train = btc_rolling[:int(0.8*len(btc_rolling))]
        test = btc_rolling[int(0.8*len(btc_rolling)):]
    
        model = ARIMA(train.values, order=arima_order)
        model_fit = model.fit(disp=0)

        fighting = np.arange(0, (test.index[-1] + forecast_days) - test.index[0])
        empty_df = pd.DataFrame(fighting)
        empty_df.index = np.arange(test.index[0], test.index[-1] + forecast_days)
    

        if ( float(0.2*len(btc_rolling)) > int(0.2*len(btc_rolling)) ):
            fc, se, conf = model_fit.forecast(len(empty_df.index), alpha=0.05)  # 95% conf
        else:
            fc, se, conf = model_fit.forecast(len(empty_df.index), alpha=0.05)

        fc_series = pd.Series(fc, index=empty_df.index)
        lower_series = pd.Series(conf[:, 0], index=empty_df.index)
        upper_series = pd.Series(conf[:, 1], index=empty_df.index)
    
        plt.rcParams.update({'font.size': 40})
        fig = plt.figure(figsize=(40,20), dpi=100)
        ax = fig.add_subplot(1,1,1)
        l1 = ax.plot(train, label = "Training")
        l2 = ax.plot(test, label = "Testing")
        l3 = ax.plot(fc_series, label = "Forecast")
        ax.fill_between(lower_series.index, upper_series, lower_series,
                         color='k', alpha=.15)
        ax.set_title('Forecast vs Actuals')
        fig.legend(loc='upper left', fontsize=40), (l1,l2,l3)
        plt.rc('grid', linestyle="-", color='black')
        plt.grid(True)
        st.write(fig)
        
#     forecast_crypto = pd.DataFrame(predict_crypto(dropdown_crypto, year_opt2))
#     forecast_crypto = forecast_crypto.T
        forecast_crypto = pd.DataFrame()
    
        f_diffed = round(len(btc_data.index)/100) * fc_series.diff()
        u_diffed = round(len(btc_data.index)/100) * upper_series.diff()
        l_diffed = round(len(btc_data.index)/100) * lower_series.diff()

        forecast_crypto["Forecasted Values"] = cumsum_shift(f_diffed, round(len(btc_data.index)/100), fc_series.values[:round(len(btc_data.index)/100)])
        forecast_crypto["Upper 95% Bound"] = cumsum_shift(u_diffed, round(len(btc_data.index)/100), upper_series.values[:round(len(btc_data.index)/100)])
        forecast_crypto["Lower 95% Bound"] = cumsum_shift(l_diffed, round(len(btc_data.index)/100), lower_series.values[:round(len(btc_data.index)/100)])
        
        date_list = [btc_data["Date"][test.index[0]] + datetime.timedelta(days=x) for x in range(len(empty_df.index))]
        forecast_crypto["Date"] = date_list
        forecast_crypto["Date"] = pd.to_datetime(forecast_crypto["Date"])
        forecast_crypto["Date"] = forecast_crypto["Date"].dt.date
        forecast_crypto.set_index("Date", inplace = True)
        return forecast_crypto
    

# This is where the user make the choice of where to reinvest the dividend paid. 

dropdown_option = st.selectbox('Where do you want to reinvest your dividends?', options)

# Create and empty DataFrame for closing prices of chosen stock
df_stock_prices = pd.DataFrame()

# Fetch the closing prices for all the stocks
df_stock_prices[dropdown_option] = close_price(dropdown_stocks)


# Calculating the projected return for reinvestment into the same stock chosen here

if dropdown_option == "Keep the cash":
    
    # Slider 3 with option to select the amount of year to reinvest(10, 20 or 30)
    year_opt3 = st.slider('How many years of pocketing the cash?', min_value= 10, max_value= 30, value=10, step= 10)
    st.write(f'You will reinvest your dividends for {year_opt3} years')
    
    daily = yf.download(dropdown_stocks, start, end)['Adj Close']
    def average_annual (daily):
        rel = daily.pct_change()
        ave_rel= rel.mean()
        anual_ret = (ave_rel * 252) * 100
        return anual_ret
    st.subheader(f'Average yearly returns of {dropdown_stocks} is {average_annual(daily): .2f}%')
    
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
    
    
    
    st.subheader(f'The projected return for {dropdown_stocks} is:')
    st.success(f'${SIP_stock_maturity}')
    
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
    
    st.subheader(f'Your total dividend return will be')
    st.success(f'${SIP_maturity}')
    

    
    # Calculating the projected return for crypto opyion chosen here
elif dropdown_option == "Same Stock":
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
    
    
    mc_stock = mc_stock_price(year_opt1)
    st.subheader(f'This is the simulated price for {dropdown_stocks} ({stock_name}).')
    st.dataframe(mc_stock)

    zero = round(mc_stock["Forecasted Average Price"][0],2)
    last = round(mc_stock["Forecasted Average Price"][year_opt1-1],2)
    pct_gain  =  ( ( (last- zero) / zero ) )

    st.info(f"The percent gain of the simulated forecasts is {round(float(pct_gain*100), 2)}%")
    
    st.subheader(f'Your stock average value after {year_opt1} years of reinvesting the dividends will be:')
    
    st.text(f"With your dividend of ${yearly_div_amount} reinvested every years, you would receive.")
    st.success(f'${round(yearly_div_amount*pct_gain,2)}')
    
    # Calculating the cumulative returns after choosing the same stock option
elif dropdown_option == "Crypto":
    
    # selection of the crypto to reinvest in
    dropdown_crypto = st.selectbox('What crypto would you like to reinvest in?', crypto)
    
    # simulation of chosen crypto using invested dividends
    
    # Getting the data for selected crypto from yahoo finance and ploting it as a line chart
    if len(dropdown_crypto) > 0:
        df = yf.download(dropdown_crypto, start, end)
        st.header('Historical value of {}'.format(dropdown_crypto))
        st.dataframe(df)
        st.line_chart(df["Adj Close"])
        st.text("Model created to forecast the moving average of crypto-currency.\nWe are using regression analysis, ")
        predict_crypto(dropdown_crypto)
    
        # Slider 2 with option to select the amount of year to reinvest(10, 20 or 30)
    year_opt2 = st.slider('Using the same regression model, how many years of investment projections?', min_value= 5, max_value= 15, value=5, step= 5)
    crypto_forecast = predict_crypto(dropdown_crypto, year_opt2)
    st.dataframe(crypto_forecast)
    
    crypto_gain = round( ((float(crypto_forecast["Forecasted Values"][-1:]) - float(crypto_forecast["Forecasted Values"][0])) / crypto_forecast["Forecasted Values"][0]) * 100 , 2)
    st.info(f"The percent gain from the forecasted values is {crypto_gain}%")
    st.text(f"Using the total yearly dividend of ${yearly_div_amount} reinvested in {dropdown_crypto} could get you:") 
    st.success(f"Future value of ${round(yearly_div_amount*crypto_gain,2)}")
       
     
   
    
#dropdown_crypto = st.selectbox('',crypto)    
#crypto_forecast = predict_crypto(dropdown_crypto, year_opt2)    
#crypto_gain = round( ((float(crypto_forecast["Forecasted Values"][-1:]) - float(crypto_forecast["Forecasted Values"][0])) / crypto_forecast["Forecasted Values"][0]) * 100 , 2)    
#crypto_future = {round(yearly_div_amount*crypto_gain,2)}      
#future_stocks = SIP_stock_maturity
#future_div = SIP_maturity


#def summary(dropdown_option):
#    senarios = ["a","b","c"]
#    for senario in senarios:
                
    

#with st.expander("See Summary of the different options"):
#    st.subheader("This shows yo uthe diffence of the 3 different options to reinvest in:")
#    col1, col2, col3 = st.columns(3)
 #   with col1:
 #       st.info("Keep the Cash")
 #       st.success(f'Stock value:${SIP_stock_maturity}')
 #       st.success(f'Dividends value:${SIP_maturity}')
 #       
 #   with col2:
 #       st.info("Same Stock")
#        st.success(f'')
        
#    with col3:
 #       st.info("Crypto")
 #       st.success(f"Future value ${round(yearly_div_amount*crypto_gain,2)}")
        
        