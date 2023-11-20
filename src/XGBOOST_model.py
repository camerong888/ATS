import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import time

# Hyperparameters:
ticker = '^GSPC'
percentage_split = 0.2

params = {
    "max_depth": [3],
    "learning_rate": [0.1], #subject to change...
    "n_estimators": [800,900],
    "colsample_bytree": [0.1], 
    "subsample": [0.8],
    "gamma": [0.5],
}

attributes = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Adj Close",
    "SMA10",
    "SMA20",
    "SMA30",
    "SMA50",
    "SMA200",
    "SMA10_derivative",
    "SMA20_derivative",
    "SMA30_derivative",
    "SMA50_derivative",
    "SMA200_derivative",
    "EMA10",
    "EMA20",
    "EMA30",
    "EMA50",
    "EMA10_derivative",
    "EMA20_derivative",
    "EMA30_derivative",
    "EMA50_derivative",
    "RSI",
    "ATR",
    "BBWidth",
    "Williams",
    "MACDs_12_26_9",
    "MACD",
    "VWAP",
    "StochasticOscillator",
    "CCI",
    "OBV",
    "ParabolicSAR",
    "AO",
    "MOM",
    "BOP",
    "RVI",
    "DMP_16",
    "DMN_16",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "STOCHk_14_3_3",
    "STOCHd_14_3_3",
    "STOCHRSIk_16_14_3_3",
    "STOCHRSId_16_14_3_3",
    "^VIX_Close",
    "^TNX_Close",
    "USO_Close",
    "XLE_Close",
    "SSE_Close",
    "Target",
]


# Load in data
file_path = f"../data/indicators/{ticker}_data_set_XGBOOST.csv"
snp500_data_set = pd.read_csv(file_path)
snp500_data_set = snp500_data_set[attributes]

# Split data
def train_test_split(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]

train, test = train_test_split(snp500_data_set, percentage_split)

X = train[:, :-1]
y = train[:, -1]

# Get best SGBOOST params
startTime = time.time()
xgbr = XGBRegressor(seed=20)
modl = GridSearchCV(
    estimator=xgbr, param_grid=params, scoring="neg_mean_squared_error", verbose=1, cv=5
)
modl.fit(X, y)
print("Best parameters:", modl.best_params_)
print("Lowest RMSE: ", (-modl.best_score_) ** (1 / 2.0))
endTime = time.time()
print(f"Parameters Fit Time: {(endTime-startTime)/60}")

model = modl.best_estimator_
model.fit(X, y)

# val = np.array(test[0, :-1]).reshape(1, -1)
# pred = model.predict(val)


def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=modl.best_params_["n_estimators"],
        colsample_bytree=modl.best_params_["colsample_bytree"],
        learning_rate=modl.best_params_["learning_rate"],
        max_depth=modl.best_params_["max_depth"],
        gamma=modl.best_params_["gamma"],
    )
    model.fit(X, y)
    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mape


def validate(data, perc):
    predictions = []
    train, test = train_test_split(data, perc)
    history = [x for x in train]

    for i in range(len(test)):
        X_test, y_test = test[i, :-1], test[i, -1]
        print(f"Attributes: {X_test}")
        print(f"Target: {y_test}") 
        pred = xgb_predict(history, X_test)
        print(f"Prediction: {pred}") 
        predictions.append(pred)
        history.append(test[i])

    error = mean_squared_error(test[:, -1], predictions, squared=False)
    MAPE = mape(test[:, -1], predictions)
    return error, MAPE, test[:, -1], predictions, test

startTime = time.time()
rmse, MAPE, y, pred, test= validate(snp500_data_set, percentage_split)
print("RMSE: " f"{rmse}")
print("MAPE: " f"{MAPE}")
endTime = time.time()
print(f"Prediction time: {(endTime-startTime)/60}")

pred = np.array(pred)

test_pred = np.c_[test, pred]
plotting_attributes = attributes + ['Pred'] 
df_TP = pd.DataFrame(test_pred, columns=plotting_attributes)



plt.figure(figsize=(15, 9))
plt.title(f"{ticker} Next Day Close Price vs. Predicted Price", fontsize=18)
plt.plot(df_TP["Target"], label="Next day Actual Closing Price", color="cyan")
plt.plot(df_TP["Pred"], label="Predicted Price", color="green", alpha=1)
plt.xlabel("Date", fontsize=18)
plt.legend(loc="upper left")
plt.ylabel("Price in USD $", fontsize=18)
plt.show()

# After fitting the model
feature_importances = model.feature_importances_
importance_series = pd.Series(feature_importances, index=snp500_data_set.columns[:-1])
importance_series_sorted = importance_series.sort_values(ascending=False)
importance_series_sorted.plot(kind="bar", title="Feature Importances")
plt.show()

n = int(len(snp500_data_set.values) * (1 - percentage_split))
snp500_data_set = snp500_data_set[n:]
snp500_data_set['Predicted'] = pred
snp500_data_set.to_csv(f'../results/{ticker}_predictions.csv', index=False)