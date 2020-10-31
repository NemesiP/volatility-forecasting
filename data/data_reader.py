########## Libaries ##########
import simfin as sf
from simfin.names import *
import quandl
from api_key import api_key


sf.set_api_key('free')
sf.set_data_dir('simfin_data/')

INC = sf.load_income(variant='quarterly', market='us')
BLC = sf.load_balance(variant='quarterly', market='us')
CSF = sf.load_cashflow(variant='quarterly', market='us')
SHP = sf.load_shareprices(variant='daily', market='us')

macro = ['CPIAUCSL', 'UNEMPLOY', 'GDP', 'PPIACO', 'DTB3', 
         'DGS10', 'INDPRO', 'EFFR', 'DCOILWTICO', 'VXTYN']

for tic in macro:
    quandl.get("FRED/"+str(tic), authtoken=api_key()).to_csv('Macroeconomic/'+str(tic)+'.csv')
    
quandl.get("RATEINF/INFLATION_USA", authtoken=api_key()).to_csv('Macroeconomic/INFLATION_USA.csv')

#CBOE Volatility Index: VIX (VIXCLS) was downloaded directly from FRED website because it couldn't be founded in Quandl
#CBOE S&P 100 Volatility Index: VXO (VXOCLS) was downloaded directly from FRED website because it couldn't be founded in Quandl
#CBOE NASDAQ 100 Volatility Index (VXNCLS) was downloaded directly from FRED website because it couldn't be founded in Quandl

