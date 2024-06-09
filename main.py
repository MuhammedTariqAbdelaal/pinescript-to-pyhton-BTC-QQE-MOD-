import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch data for Bitcoin (BTC-USD)
btc_data = yf.download('BTC-USD', start='2020-01-01', end='2023-12-31', interval='1d')


# Helper functions
def rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# Inputs
RSI_Period = 6
SF = 5
QQE = 3
ThreshHold = 3

# Calculations
src = btc_data['Close']
Wilders_Period = RSI_Period * 2 - 1
Rsi = rsi(src, RSI_Period)
RsiMa = ema(Rsi, SF)
AtrRsi = abs(RsiMa.shift(1) - RsiMa)
MaAtrRsi = ema(AtrRsi, Wilders_Period)
dar = ema(MaAtrRsi, Wilders_Period) * QQE

# Initialize bands and trend
longband = pd.Series(0, index=btc_data.index)
shortband = pd.Series(0, index=btc_data.index)
trend = pd.Series(0, index=btc_data.index)

DeltaFastAtrRsi = dar
RSIndex = RsiMa
newshortband = RSIndex + DeltaFastAtrRsi
newlongband = RSIndex - DeltaFastAtrRsi

# Band calculations
for i in range(1, len(btc_data)):
    longband[i] = (RSIndex[i - 1] > longband[i - 1] and RSIndex[i] > longband[i - 1]) * max(longband[i - 1],
                                                                                            newlongband[i]) + \
                  (RSIndex[i - 1] <= longband[i - 1] or RSIndex[i] <= longband[i - 1]) * newlongband[i]
    shortband[i] = (RSIndex[i - 1] < shortband[i - 1] and RSIndex[i] < shortband[i - 1]) * min(shortband[i - 1],
                                                                                               newshortband[i]) + \
                   (RSIndex[i - 1] >= shortband[i - 1] or RSIndex[i] >= shortband[i - 1]) * newshortband[i]

cross_1 = (longband.shift(1) < RSIndex) & (RSIndex <= longband)
trend = (RSIndex < shortband.shift(1)) * 1 - (RSIndex >= shortband.shift(1)) * cross_1 + trend.shift(1).fillna(1)
FastAtrRsiTL = (trend == 1) * longband + (trend == -1) * shortband

# Bollinger Bands
length = 50
mult = 0.35
basis = FastAtrRsiTL.rolling(window=length).mean() - 50
dev = mult * FastAtrRsiTL.rolling(window=length).std()
upper = basis + dev
lower = basis - dev
color_bar = np.where(RsiMa - 50 > upper, '#00c3ff', np.where(RsiMa - 50 < lower, '#ff0062', 'gray'))

# Zero cross calculations
QQEzlong = (RSIndex >= 50).cumsum()
QQEzshort = (RSIndex < 50).cumsum()

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index, FastAtrRsiTL - 50, label='QQE Line', color='white', linewidth=2)
plt.plot(btc_data.index, RsiMa - 50, label='Histo2', color='silver', alpha=0.5)

Greenbar1 = RsiMa - 50 > ThreshHold
Greenbar2 = RsiMa - 50 > upper
Redbar1 = RsiMa - 50 < -ThreshHold
Redbar2 = RsiMa - 50 < lower

plt.bar(btc_data.index, (Greenbar1 & Greenbar2) * (RsiMa - 50), label='QQE Up', color='#00c3ff')
plt.bar(btc_data.index, (Redbar1 & Redbar2) * (RsiMa - 50), label='QQE Down', color='#ff0062')

plt.title('QQE MOD')
plt.legend()
plt.show()
