import yfinance as yf
import os
from datetime import datetime, timedelta

tS = os.environ['Stock_Name']
print()
print()
print(tS)


getDateTime = datetime.now() + timedelta(1)
today = datetime.now()
today = datetime.strftime(today, '%Y-%m-%d')
DateTime = datetime.strftime(getDateTime, '%Y-%m-%d')
print(today)

#define the ticker symbol
tickerSymbol = tS
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start = DateTime, end = DateTime)


output_path = '{}.csv'.format(tickerSymbol)

try:
    tickerDf.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    print("get today's stock price - {}".format(tS))
except:
    tickerDf.reset_index().to_csv('{}.csv'.format(tickerSymbol), mode='a', index = False)
