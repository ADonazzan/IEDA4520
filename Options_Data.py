import yfinance as yf # https://github.com/ranaroussi/yfinance
import pandas as pd
aapl = yf.Ticker("AAPL")

print(aapl.options) # list of dates 
#DF_calls = aapl.option_chain(aapl.options[1])
#DF_calls = pd.DataFrame(DF_calls[1])

