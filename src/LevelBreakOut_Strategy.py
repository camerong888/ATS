import pandas as pd
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import yfinance as yf

# Hyperparameters for Price Breakout Strategy helper function for DataGenerator
ticker = '^GSPC'
backcandles = 40
window=6

def PriceBreakoutStrategy(df, backcandles, window, showGraph = False):

    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)

    df=df[0:]
    df.reset_index(drop=True, inplace=True)

    EMAsignal = [0]*len(df)

    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if max(df.open[i], df.close[i])>=df.EMA50[i]:
                dnt=0
            if min(df.open[i], df.close[i])<=df.EMA50[i]:
                upt=0
        if upt==1 and dnt==1:
            EMAsignal[row]=3
        elif upt==1:
            EMAsignal[row]=2
        elif dnt==1:
            EMAsignal[row]=1

    df['EMASignal'] = EMAsignal

    def isPivot(candle, window):
        """
        function that detects if a candle is a pivot/fractal point
        args: candle index, window before and after candle to test if pivot
        returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
        """
        if (candle - window) < 0 or (candle + window) >= len(df):
            return 0
        
        pivotHigh = 1
        pivotLow = 2
        for i in range(candle-window, candle+window+1):
            if df.iloc[candle].low > df.iloc[i].low:
                pivotLow=0
            if df.iloc[candle].high < df.iloc[i].high:
                pivotHigh=0
        if (pivotHigh and pivotLow):
            return 3
        elif pivotHigh:
            return pivotHigh
        elif pivotLow:
            return pivotLow
        else:
            return 0

    df['isPivot'] = df.apply(lambda x: isPivot(x.name,window), axis=1)

    # print(df[df['isPivot']!=0])

    def pointpos(x):
        if x['isPivot']==2:
            return x['low']-1e-3
        elif x['isPivot']==1:
            return x['high']+1e-3
        else:
            return np.nan
        
    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

    if showGraph:
        dfpl = df
        fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                        open=dfpl['open'],
                        high=dfpl['high'],
                        low=dfpl['low'],
                        close=dfpl['close'])])

        fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                        marker=dict(size=5, color="MediumPurple"),
                        name="pivot")
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    def detect_structure(candle, backcandles, window):

        if (candle <= (backcandles+window)) or (candle+window+1 >= len(df)):
            return 0
            
        localdf = df.iloc[candle-backcandles-window:candle-window] #window must be greater than pivot window to avoid look ahead bias
        highs = localdf[localdf['isPivot'] == 1].high.tail(3).values
        lows = localdf[localdf['isPivot'] == 2].low.tail(3).values
        levelbreak = 0
        zone_width = 0.01
        if len(lows)==2:
            support_condition = True
            mean_low = lows.mean()
            for low in lows:
                if abs(low-mean_low)>zone_width:
                    support_condition = False
                    break
            if support_condition and (mean_low - df.loc[candle].close)>zone_width*2:
                levelbreak = 1

        if len(highs)==2:
            resistance_condition = True
            mean_high = highs.mean()
            for high in highs:
                if abs(high-mean_high)>zone_width:
                    resistance_condition = False
                    break
            if resistance_condition and (df.loc[candle].close-mean_high)>zone_width*2:
                levelbreak = 2
        return levelbreak

    df['pattern_detected'] = df.apply(lambda row: detect_structure(row.name, backcandles=backcandles, window=window), axis=1)
    
    # print(df[df['pattern_detected']!=0])

    # print(df[df["isPivot"]!=0])
    # print(df[df["pointpos"].notna()])

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    # print(df)

    return df[['EMASignal', 'isPivot', 'pattern_detected']]


# result = PriceBreakoutStrategy(df=df, backcandles=backcandles, window=window)

# print(result[result['pattern_detected']!=0])


