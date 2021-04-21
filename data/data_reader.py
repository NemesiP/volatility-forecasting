########## Libaries ##########
import simfin as sf
from simfin.names import *
import quandl
from api_key import quandl_api_key
import pandas as pd
from datetime import datetime
import yfinance as yf
import time
from os import path
"""
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
"""
# Getting all the symbol's tickers name from Barchart
tic = pd.read_csv('sp-500-index-11-01-2020.csv')
ticker = tic.Symbol[:-1]
ticker = [tc.replace('.', '-') for tc in ticker]
start = datetime(1997,1,1)
end = datetime(2020,11,1)

# Checking if the stockprices have been downloaded or not. If they haven't been, then it will be.
for i in ticker:
#    if path.exists('Stocks/{}.csv'.format(i)):
#        print('{} has been downloaded previously!'.format(i))
#    else:
    time.sleep(1)
    df = yf.download(i, start = start, end = end)
    df.to_csv('C:/Users/peter/Desktop/volatility-forecasting/data/stocks/{}.csv'.format(i))