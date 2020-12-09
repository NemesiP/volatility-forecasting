import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FeaturesManipulation:
    def select_timeframe(df, start, end = None):
        try:
            df.Date = pd.to_datetime(df.Date)
        except:
            df.Date = pd.to_datetime(df.DATE)
        if end == None:
            df = df[(df.Date >= start) & (df.Date < datetime.today())]
        else:
            df = df[(df.Date >= start) & (df.Date < end)]
        df = df.reset_index(drop = True)
        return df
    
    def compress_data(df1, df2, col, lag = None):
        per = df1.Date.dt.to_period('M')
        uniq = np.array(per.unique())
        per1 = df2.Date.dt.to_period('M')
        dd = {}
        for i in range(len(uniq)-1):
            index = per1[(per1 >= uniq[i].strftime('%Y-%m')) & (per1 < uniq[i+1].strftime('%Y-%m'))].index.values
            arr = []
            if lag == None:
                for j in range(len(index) - len(index), len(index)):
                    try:
                        arr.append(df2[col][df2.index == index[j]].values[0])
                    except:
                        arr.append(0.0)
            elif lag > len(index):
                print('Your lag is out of range! Please give appropriate lag value.')
                break
            else:
                for j in range(len(index) - lag, len(index)):
                    try:
                        arr.append(df2[col][df2.index == index[j]].values[0])
                    except:
                        arr.append(0.0)
    
            try:
                dd[uniq[i+1].strftime('%Y-%m')] = arr
            except:
                pass
        return pd.DataFrame(dd).T
    
    def build_lag(s, lag=2, dropna=True):
        if type(s) is pd.DataFrame:
            new_dict = {}
            for col_name in s:
                new_dict[col_name] = s[col_name]
                for l in range(1, lag + 1):
                    new_dict['%s_lag%d' %(col_name, l)]=s[col_name].shift(l)
            res = pd.DataFrame(new_dict, index = s.index)
            
        elif type(s) is pd.Series:
            the_range = range(lag+1)
            res = pd.concat([s.shift(i) for i in the_range], axis = 1)
            res.columns = ['lag_%d' %i for i in the_range]
        else:
            print('Only works for DataFrame or Series')
            return None
        if dropna:
            return res.dropna()
        else:
            return res
    

if __name__ == '__main__':
    print('This module is an additional file for modelling!')