# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:59:29 2021

@author: peter
"""
import pandas as pd

def create_lagged_variable(data, name, lag):
    """
    This function is about to create the lagged variables.

    Parameters
    ----------
    data : DataFrame
        Pandas dataframe that contains one regressor.

    Returns
    -------
    new_df : DataFrame
        Pandas dataframe that contains all the lagged variables about that one regressor.

    """
    ## Létrehozok egy új dataframe-et, amibe fogom behelyzeni a késleltett értékeit az magyarázóváltozómnak.
    new_df = pd.DataFrame(data = {name: data})
    for i in range(lag):
        ## Annyi késleltetést készítek, amennyit a self.lag-ban megadtam
        new_df['Lag {number}'.format(number = i + 1)] = new_df[name].shift(i + 1).fillna(0)
    return new_df
    
def create_matrix(data, lag):
    """
    This function is a helper function for the weight functions.
    It puts all the lagged variables into a dictionary.

    Parameters
    ----------
    data : DataFrame
        Pandas dataframe that contains all the regressors.

    Returns
    -------
    X : Dictionary
        Dictionary that contains all the lagged regressors.

    """
    ## Létrehozok egy dictionary-t, amibe fognak kerülni a magyarázóváltozóim késletetéseinek mátrixa.
    X = {}
    
    ## Annyiszor végezzük el a ciklust, ahány oszlopa van a dataframe-nek.
    for i in range(1, len(data.columns) + 1):
        ## A dictionary-be belehelyzem ezeket a mátrixokat, és létrehozok hozzájuk egy kulcsot [X1, ..., Xn], amikkel megtudom hívni ezeket.
        X['X{num}'.format(num = i)] = create_lagged_variable(data.iloc[:, i - 1], data.columns[i - 1], lag).iloc[:, -lag:].values
    return X