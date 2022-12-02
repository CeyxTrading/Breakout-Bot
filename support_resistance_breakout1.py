import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
from backtesting import Strategy
from backtesting import Backtest
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import ADXIndicator


"""
Sources: 
https://github.com/Chiu-Huang/Support-Resistance-Line-Algo (MIT License)
https://medium.com/@huangchingchiu/generate-support-resistance-line-for-any-stocks-in-3-minutes-871049515353
Credited to Mott The Tuple @ https://stackoverflow.com/questions/8587047/support-resistance-algorithm-technical-analysis/55311893#55311893
"""

OUTPUT_DIR = "C:\\dev\\trading\\tradesystem1\\results\\plots\\"


def fetch_price_data(symbol):
    #  valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    period = '5y'
    #  valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    interval = '1d'
    data = yf.download(tickers=symbol, period=period, interval=interval)
    price_df = pd.DataFrame(data)
    price_df = price_df.tail(600)
    price_df.reset_index(inplace=True)
    return price_df


def calculate_support_resistance(df, rolling_wave_length, num_clusters):
    date = df.index

    # Reset index for merging
    df.reset_index(inplace=True)

    # Create min and max waves
    max_waves_temp = df.High.rolling(rolling_wave_length).max().rename('waves')
    min_waves_temp = df.Low.rolling(rolling_wave_length).min().rename('waves')

    max_waves = pd.concat([max_waves_temp, pd.Series(np.zeros(len(max_waves_temp)) + 1)], axis=1)
    min_waves = pd.concat([min_waves_temp, pd.Series(np.zeros(len(min_waves_temp)) + -1)], axis=1)

    #  Remove dups
    max_waves.drop_duplicates('waves', inplace=True)
    min_waves.drop_duplicates('waves', inplace=True)

    #  Merge max and min waves
    waves = max_waves.append(min_waves).sort_index()
    waves = waves[waves[0] != waves[0].shift()].dropna()

    # Find Support/Resistance with clustering using the rolling stats
    # Create [x,y] array where y is always 1
    x = np.concatenate((waves.waves.values.reshape(-1, 1),
                        (np.zeros(len(waves)) + 1).reshape(-1, 1)), axis=1)

    # Initialize Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    cluster.fit_predict(x)
    waves['clusters'] = cluster.labels_

    # Get index of the max wave for each cluster
    waves2 = waves.loc[waves.groupby('clusters')['waves'].idxmax()]
    df.index = date

    waves2.waves.drop_duplicates(keep='first', inplace=True)

    return waves2.reset_index().waves


def plot_chart(index, df, support_resistance_levels, adx, adx_pos, adx_neg):
    light_palette = {}
    light_palette["bg_color"] = "#ffffff"
    light_palette["plot_bg_color"] = "#ffffff"
    light_palette["grid_color"] = "#e6e6e6"
    light_palette["text_color"] = "#2e2e2e"
    light_palette["dark_candle"] = "#4d98c4"
    light_palette["light_candle"] = "#b1b7ba"
    light_palette["volume_color"] = "#c74e96"
    light_palette["border_color"] = "#2e2e2e"
    palette = light_palette

    #  Array of colors for support/resistance lines
    support_resistance_colors = ["#5c285b", "#802c62", "#a33262", "#c43d5c", "#de4f51","#f26841", "#fd862b", "#ffa600","#3366d6"]

    #  Create sub plots
    fig = make_subplots(rows=3, cols=1, subplot_titles=[f"{symbol} Chart", "ADX Trend Strength", "ADX Direction Pos/Neg"], \
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]], \
                        vertical_spacing=0.04, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])

    #  Plot close price
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 close=df['Close'],
                                 low=df['Low'],
                                 high=df['High'],
                                 increasing_line_color=palette['light_candle'],
                                 decreasing_line_color=palette['dark_candle'], name='Price'), row=1, col=1)

    #  Add Support/Resistance levels
    i = 0
    for level in support_resistance_levels.to_list():
        line_color = support_resistance_colors[i] if i < len(support_resistance_colors) else support_resistance_colors[0]
        fig.add_hline(y=level, line_width=1, line_dash="dash", line_color=line_color, row=1, col=1)
        i += 1

    #  ADX trend strength
    fig.add_trace(go.Line(name='ADX Trend Strength', x=df.index, y=adx, marker_color="blue"), row=2, col=1)

    #  ADX trend direction
    fig.add_trace(go.Line(name='+DI', x=df.index, y=adx_pos, marker_color="blue"), row=3, col=1)
    fig.add_trace(go.Line(name='-DI', x=df.index, y=adx_neg, marker_color="red"), row=3, col=1)

    fig.update_layout(
        title={'text': '', 'x': 0.5},
        font=dict(family="Verdana", size=12, color=palette["text_color"]),
        autosize=True,
        width=1280, height=720,
        xaxis={"rangeslider": {"visible": False}},
        plot_bgcolor=palette["plot_bg_color"],
        paper_bgcolor=palette["bg_color"])
    fig.update_yaxes(visible=False, secondary_y=True)
    #  Change grid color
    fig.update_xaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"])
    fig.update_yaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"])

    #  Save plot
    file_name = f"{index}_support_resistance_adx.png"
    path = os.path.join(OUTPUT_DIR, file_name)
    fig.write_image(path, format="png")


def ADX(df, window):
    indicator_adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
    return indicator_adx.adx()


def ADX_NEG(df, window):
    indicator_adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
    return indicator_adx.adx_neg()


def ADX_POS(df, window):
    indicator_adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
    return indicator_adx.adx_pos()


class MixedStrategy(Strategy):
    adx_window = 14
    adx_strength_threshold = 20

    long_hold = 0
    long_entry_price = 0

    stop_loss_pc = 5
    trailing_stop_pc = 1

    long_peak_price = 0
    last_purchase_price = 0

    i = 0

    def init(self):
        super().init()

        # Add indicators
        self.adx = self.I(ADX, self.data.df, self.adx_window)
        self.adx_pos_neg = self.I(lambda *args: (ADX_POS(*args), ADX_NEG(*args)),self.data.df, self.adx_window)

    def next(self):
        super().init()

        self.i += 1

        long_entry_signal = 0
        long_exit_signal = 0


        #  LONG ENTRY
        #----------------------------------------------

        #  calculate support/resistance levels
        rolling_wave_length = 20
        num_clusters = 4

        if self.i < 200 + rolling_wave_length:
            return

        #  Grab a subset of the data
        support_resistance_lb = self.data.df.iloc[-200 - rolling_wave_length:]

        #  Calculate support/resistance
        support_resistance_levels = calculate_support_resistance(support_resistance_lb, rolling_wave_length, num_clusters)

        #  Check resistance crossover
        current_price = self.data.Close[-1]
        for level in support_resistance_levels.to_list():
            if self.data.Close[-1] >= level and self.data.Close[-2] <= level:
                long_entry_signal += 1
                break

        #  ADX trend strength
        if self.adx[-1] >= self.adx_strength_threshold:
            long_entry_signal += 1
        
        #  Check trend direction
        adx_pos = self.adx_pos_neg[0]
        adx_neg = self.adx_pos_neg[1]
        if adx_pos[-1] > adx_neg[-1]:
            long_entry_signal += 1

        #  Plot chart
        if self.long_hold == 0 and long_entry_signal >= 3:
            plot_chart(self.i, self.data.df, support_resistance_levels, self.adx, adx_pos, adx_neg)

        #  LONG EXIT
        #----------------------------------------------
        #  Stop loss
        if self.long_hold == 1 and current_price <= (self.last_purchase_price * (1 - (self.stop_loss_pc/100))):
            long_exit_signal += 1

        #  Track max price after long entry
        if self.long_hold == 1 and current_price > self.long_peak_price:
            self.long_peak_price = current_price

        #  Trailing stop loss
        trailing_stop_price = self.long_peak_price * (1 - (self.trailing_stop_pc/100))
        if self.long_hold == 1 and current_price <= trailing_stop_price:
            long_exit_signal += 1

        #  Long entry
        if self.long_hold == 0 and long_entry_signal >= 3:
            #  Buy
            self.buy()
            self.long_hold = 1
            self.last_purchase_price = current_price
            self.long_peak_price = current_price

        # Long exit
        elif self.long_hold == 1 and long_exit_signal >= 1:
            self.position.close()
            self.long_hold = 0


def run_backtest(df):
    # If exclusive orders (each new order auto-closes previous orders/position),
    # cancel all non-contingent orders and close all open trades beforehand
    bt = Backtest(df, MixedStrategy, cash=100000, commission=0.00075, trade_on_close=True, exclusive_orders=True, hedging=False)
    stats = bt.run()
    print(stats)
    bt.plot()


# MAIN
if __name__ == '__main__':
    procs = []

    symbol = 'TSLA'
    interval = '1day'
    start_date_str = '2012-01-01'
    end_date_str = '2022-09-30'
    resampleFreq = '1Day'

    #  Get prices
    df = fetch_price_data(symbol)

    #  Run backtest
    run_backtest(df)


