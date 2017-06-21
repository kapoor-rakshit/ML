import pandas as pd
import quandl

df=quandl.get("BSE/BOM507685",authtoken="Yiu9TpqKozssyt4ja9tA",start_date="2017-01-01", end_date="2017-02-10")    
# wipro BSE code 
# https://www.quandl.com/data/NSE-National-Stock-Exchange-of-India    for more company stock codes
# https://docs.quandl.com/docs/parameters-2                           for params to be filtered

print(df)