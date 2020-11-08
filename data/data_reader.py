########## Libaries ##########
import simfin as sf
from simfin.names import *
import quandl
from api_key import quandl_api_key, alpha_api_key
import csv
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from datetime import datetime
import time
from os import path

########## Downloading the fundamental data from SimFin ##########
sf.set_api_key('free')
sf.set_data_dir('simfin_data/')

INC = sf.load_income(variant='quarterly', market='us')
BLC = sf.load_balance(variant='quarterly', market='us')
CSF = sf.load_cashflow(variant='quarterly', market='us')

########## Downloading the macroeconomic data from Quandl (FRED, Rateinf) ##########
macro = ['CPIAUCSL', 'UNEMPLOY', 'GDP', 'PPIACO', 'DTB3', 
         'DGS10', 'INDPRO', 'EFFR', 'DCOILWTICO', 'VXTYN']

for tic in macro:
    if path.exists('Macroeconomic/'+str(tic)+'.csv') == True:
        print(str(tic)+' data have been downloaded previously!')
    else:
        quandl.get("FRED/"+str(tic), authtoken=quandl_api_key()).to_csv('Macroeconomic/'+str(tic)+'.csv')
        print(str(tic)+' data downloaded!')

if path.exists('Macroeconomic/INFLATION_USA.csv') == True:
    print('INFLATION_USA data have been downloaded previously!')
else:
    quandl.get("RATEINF/INFLATION_USA", authtoken=quandl_api_key()).to_csv('Macroeconomic/INFLATION_USA.csv')
    print('INFLATION_USA data downloaded!')

#CBOE Volatility Index: VIX (VIXCLS) was downloaded directly from FRED website because it couldn't be founded in Quandl
#CBOE S&P 100 Volatility Index: VXO (VXOCLS) was downloaded directly from FRED website because it couldn't be founded in Quandl
#CBOE NASDAQ 100 Volatility Index (VXNCLS) was downloaded directly from FRED website because it couldn't be founded in Quandl

########### Downloading the daily prices data from Alpha Vantage ##########
def daily_downloader(tic, start, end, API):
    web.DataReader(str(tic), 'av-daily', start = start, end = end, api_key = API).to_csv('Stocks/'+str(tic)+'.csv')
    return print(str(tic)+' symbol data downloaded!')

def get_stock_prices(tic, start, end, API):
    for tc in ticker:
        if path.exists('Stocks/'+str(tc)+'.csv') == True:
            print(str(tc)+' have been downloaded previously!')
        else:
            daily_downloader(tc.replace('.','-'), start, end, API) # In the function I had to add '.replace('.','-') because in this list the '-' was written as '.'
            time.sleep(12) # Time must have been added due to the limitation of Alpha Vantage 5 calls per minute
            
# Getting all the symbol's tickers name from Barchart
tic = pd.read_csv('sp-500-index-11-01-2020.csv')
ticker = tic.Symbol[:-1]
start = datetime(2000,1,1)
end = datetime(2020,11,1)

# Checking if the stock prices have been downloaded or not. If they haven't been, then it will be.
get_stock_prices(ticker, start, end, alpha_api_key())