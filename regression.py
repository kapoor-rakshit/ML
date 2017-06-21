import pandas as pd
import quandl                    # quandl.com is financial dataset site 
from math import *

dataframe=quandl.get("nse/wipro",authtoken="Yiu9TpqKozssyt4ja9tA",start_date="2017-01-01",end_date="2017-02-05",limit=10)

# wipro NSE code 
# https://www.quandl.com/data/NSE-National-Stock-Exchange-of-India    for more company stock codes
# https://docs.quandl.com/docs/parameters-2                           for params to be filtered

# limit=5 displays the last 5 records from end_date, default end_date is current date

""" print(dataframe) """
# each column is a feature

dataframe["HLChange"]=(dataframe["High"]-dataframe["Low"])/(dataframe["Close"])*100     # adds new features (column)
dataframe["COChange"]=(dataframe["Close"]-dataframe["Open"])/(dataframe["Open"])*100

dataframe=dataframe[["HLChange","COChange","Open","Close"]]                    # Choose which features to use

forecast_on="Close"                                         # feature to predict value on
dataframe.fillna(-99999,inplace=True)                       # if data is NaN fill it with -99999
forecast_for_next=int(ceil(0.3*len(dataframe)))             # if data for 100 days then calculate for (0.3) 30% days
dataframe["FutureClose"]=dataframe[forecast_on].shift(-forecast_for_next)   # value in next 30 days
dataframe.dropna(inplace=True)                                              # if any row has any feature as NaN drop that



print(dataframe)