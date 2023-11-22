import backtrader as bt
import pandas as pd
import numpy as np
import math
import random


class PandasDataWithPredictions(bt.feeds.PandasData):
    lines = (
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "sma10",
        "sma20",
        "sma30",
        "sma50",
        "sma200",
        "sma10_derivative",
        "sma20_derivative",
        "sma30_derivative",
        "sma50_derivative",
        "sma200_derivative",
        "ema10",
        "ema20",
        "ema30",
        "ema50",
        "ema10_derivative",
        "ema20_derivative",
        "ema30_derivative",
        "ema50_derivative",
        "rsi",
        "atr",
        "bbwidth",
        "williams",
        "macd",
        "vwap",
        "stochasticoscillator",
        "cci",
        "obv",
        "parabolicsar",
        "ao",
        "mom",
        "bop",
        "rvi",
        "dmp_16",
        "dmn_16",
        "macd_12_26_9",
        "macdh_12_26_9",
        "macds_12_26_9",
        "stochk_14_3_3",
        "stochd_14_3_3",
        "stochrsik_16_14_3_3",
        "stochrsid_16_14_3_3",
        "vix_close",
        "tnx_close",
        "uso_close",
        "xle_close",
        "sse_close",
        "emasignal",
        "ispivot",
        "pattern_detected",
        "target",
        "predicted",
    )
    params = (
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("adj_close", -1),
        ("volume", -1),
        ("sma10", -1),
        ("sma20", -1),
        ("sma30", -1),
        ("sma50", -1),
        ("sma200", -1),
        ("sma10_derivative", -1),
        ("sma20_derivative", -1),
        ("sma30_derivative", -1),
        ("sma50_derivative", -1),
        ("sma200_derivative", -1),
        ("ema10", -1),
        ("ema20", -1),
        ("ema30", -1),
        ("ema50", -1),
        ("ema10_derivative", -1),
        ("ema20_derivative", -1),
        ("ema30_derivative", -1),
        ("ema50_derivative", -1),
        ("rsi", -1),
        ("atr", -1),
        ("bbwidth", -1),
        ("williams", -1),
        ("macd", -1),
        ("vwap", -1),
        ("stochasticoscillator", -1),
        ("cci", -1),
        ("obv", -1),
        ("parabolicsar", -1),
        ("ao", -1),
        ("mom", -1),
        ("bop", -1),
        ("rvi", -1),
        ("dmp_16", -1),
        ("dmn_16", -1),
        ("macd_12_26_9", -1),
        ("macdh_12_26_9", -1),
        ("macds_12_26_9", -1),
        ("stochk_14_3_3", -1),
        ("stochd_14_3_3", -1),
        ("stochrsik_16_14_3_3", -1),
        ("stochrsid_16_14_3_3", -1),
        ("vix_close", -1),
        ("tnx_close", -1),
        ("uso_close", -1),
        ("xle_close", -1),
        ("sse_close", -1),
        ("emasignal", -1),
        ("ispivot", -1),
        ("pattern_detected", -1),
        ("target", -1),
        ("predicted", -1),
    )


# Custom Strategy using the loaded XGBoost model
class XGBStrategy(bt.Strategy):
    params = dict(
        price_change_threshold=0.1,
        stop_loss=0.1,
        take_profit=0.1,
        half_take_profit=0.1 / 2,
        risk_percent=0.999,
    )

    def __init__(self):
        self.data_predicted = self.datas[0].close  # Placeholder for predicted values
        self.order = None
        self.buyprice = None
        self.half_sold = False
        self.bought = False
        self.order_size = 0

    def next(self):
        # if self.order:
        #     return

        prediction = self.datas[0].predicted[0]
        current_price = self.datas[0].adj_close[0]

        # Determine whether the predicted price is significantly higher than the current price
        if not self.bought:  # if not in the market
            if prediction > current_price * (1.0 + self.params.price_change_threshold):
                stop_price = current_price * (1.0 - self.params.stop_loss)
                self.order_size = self.calculate_order_size(stop_price)
                self.order = self.buy(size=self.order_size)
                self.buyprice = current_price  # Store the price at which we bought
                self.half_sold = False
                self.bought = True
                print("Bought")
        else:
            if (
                self.bought
                and not self.half_sold
                and current_price > self.buyprice * (1.0 + self.params.half_take_profit)
            ):
                # print(f"Broker cash before sell: {self.broker.get_cash()}")
                self.sell(size=self.order_size / 2)  # Sell half the position
                self.half_sold = True
                print("Sold half at partial profit")
            elif (
                self.bought
                and not self.half_sold
                and current_price < self.buyprice * (1.0 - self.params.stop_loss)
            ):
                # print(f"Broker cash before sell: {self.broker.get_cash()}")
                self.sell(size=self.order_size)  # Sell the position
                self.bought = False
                print("Sold at full loss")

            if self.bought and self.half_sold and current_price < self.buyprice:
                # print(f"Broker cash before sell: {self.broker.get_cash()}")
                self.sell(size=self.order_size / 2)  # Full sell at adjusted stop loss
                self.bought = False
                print("Sold at adjusted stop loss")
            elif (
                self.bought
                and self.half_sold
                and current_price > self.buyprice * (1.0 + self.params.take_profit)
            ):
                # print(f"Broker cash before sell: {self.broker.get_cash()}")
                self.sell(size=self.order_size / 2)  # Full sell at take profit
                self.bought = False
                print("Sold at full profit")

    def calculate_order_size(self, stop_price):
        # print(f"Broker cash: {self.broker.get_cash()}")
        risked_value = self.broker.get_cash() * self.params.risk_percent
        # print(f"Risked value: {risked_value}")
        position_size = risked_value / (self.datas[0].adj_close[0])
        position_size = math.floor(position_size)
        # Ensure that position size is at least 1 if risked value allows
        if position_size == 0 and risked_value > 0:
            position_size = 1
            # print("Adjusted position size to minimum of 1 due to rounding.")
        else:
            # print(f"Position size: {position_size}")
            pass
        return position_size


def run_backtest(
    global_price_change_threshold,
    global_profit_loss_ratio,
    global_take_profit,
    global_risk_percentage,
    initialInvestment,
    showPlot = False
):
    # Initialize the backtrader engine
    cerebro = bt.Cerebro()

    # Add the data feed
    cerebro.adddata(data)

    # Add the strategy
    cerebro.addstrategy(
        XGBStrategy,
        price_change_threshold=global_price_change_threshold,
        stop_loss=global_take_profit / global_profit_loss_ratio,
        take_profit=global_take_profit,
        risk_percent=global_risk_percentage,
    )

    # Set the initial capital
    cerebro.broker.setcash(initalInvestment)

    # Set the trade size
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Run the backtest
    backtest_result = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(
        f"Final Portfolio Value: {final_value} for Threshold: {global_price_change_threshold}, Take Profit: {global_take_profit}, Profit Loss Ratio: {global_profit_loss_ratio}"
    )

    # Plot the results
    if showPlot:
        cerebro.plot()

    return final_value


# Initial Parameters
ticker = "^GSPC"
percentage_split = 0.2
initalInvestment = 100000
runLoop = False

# Load full data to extract dates
file_path = f"../data/indicators/{ticker}_data_set_XGBOOST.csv"
full_data = pd.read_csv(file_path)
n = int(len(full_data) * (1 - percentage_split))
split_data = full_data[n:]
backtest_data_with_predictions = pd.read_csv(f"../results/{ticker}_predictions.csv")
backtest_data_with_predictions["Date"] = split_data["Date"].values  # Align dates
backtest_data_with_predictions["Date"] = pd.to_datetime(
    backtest_data_with_predictions["Date"]
)
backtest_data_with_predictions.set_index("Date", inplace=True)
required_columns = ["Open", "High", "Low", "Close", "Volume"]
for col in required_columns:
    if col not in backtest_data_with_predictions.columns:
        print(col)
        backtest_data_with_predictions[col] = split_data[col].values
data = PandasDataWithPredictions(
    dataname=backtest_data_with_predictions, adj_close="Adj Close"
)

if runLoop:
    # Loop to find optimal parameters
    results = []
    numLoops = 500
    for member in range(numLoops):
        global_risk_percentage = 0.999
        global_price_change_threshold = random.uniform(
            0.001, 0.01
        )  # Random value for testing
        global_take_profit = random.uniform(0.01, 0.2)  # Random value for testing
        global_profit_loss_ratio = random.uniform(1, 2.5)

        final_value = run_backtest(
            global_price_change_threshold,
            global_profit_loss_ratio,
            global_take_profit,
            global_risk_percentage,
            initalInvestment,
        )

        results.append(
            {
                "price_change_threshold": global_price_change_threshold,
                "profit_loss_ratio": global_profit_loss_ratio,
                "take_profit": global_take_profit,
                "risk_percent": global_risk_percentage,
                "final_value": final_value,
            }
        )

    sorted_results = sorted(results, key=lambda x: x["final_value"], reverse=True)

    # Print out the top 5 results
    print("Top 5 Backtest Results:")
    for result in sorted_results[:5]:
        print(
            f"Threshold: {result['price_change_threshold']:.4f}, "
            f"Profit Loss Ratio: {result['profit_loss_ratio']:.4f}, "
            f"Take Profit: {result['take_profit']:.4f}, "
            f"Risk: {result['risk_percent']:.4f}, "
            f"Final Value: {result['final_value']:.2f}"
        )
else:
    global_risk_percentage = 0.999
    global_price_change_threshold = 0.0075
    global_take_profit = 0.02
    global_profit_loss_ratio = 2.0
    final_value = run_backtest(
        global_price_change_threshold,
        global_profit_loss_ratio,
        global_take_profit,
        global_risk_percentage,
        initalInvestment,
        showPlot = True
    )



# support and resistance and engulfing pattern
# Last 4-8 weeks
# automate it to have parameters update every sunday for next week
# classify trends using moving averages, candles above or below moving average
# 5 minutes is too noisy too difficult
# 4 hours and daily timeframes are optimal
# Do not fully automate, send send indicators to email
# if anything important is happening in the news later in the day avoid trading


# Threshold: 0.0062, Profit: 0.0893, Loss: 0.0999, Risk: 0.9990, Final Value: 183734.35
# Threshold: 0.0083, Profit: 0.0662, Loss: 0.0597, Risk: 0.9990, Final Value: 179757.14
# Threshold: 0.0016, Profit: 0.0538, Loss: 0.1505, Risk: 0.9990, Final Value: 178642.24
# Threshold: 0.0060, Profit: 0.0905, Loss: 0.1188, Risk: 0.9990, Final Value: 177521.56
# Threshold: 0.0055, Profit: 0.0248, Loss: 0.0782, Risk: 0.9990, Final Value: 177276.24
