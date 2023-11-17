import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Hyperparameters:
params = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [500, 700, 1000],
    "colsample_bytree": [0.1, 0.3, 0.5],
    "subsample": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5],
}

file_path = "../data/indicators/snp500_data_set_XGBOOST.csv"
snp500_data_set = pd.read_csv(file_path)

snp500_data_set = snp500_data_set.drop(columns=["Date"])


def train_test_split(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]


train, test = train_test_split(snp500_data_set, 0.2)

X = train[:, :-1]
y = train[:, -1]

# print(X)
# print(y)

xgbr = XGBRegressor(seed=20)
modl = GridSearchCV(
    estimator=xgbr, param_grid=params, scoring="neg_mean_squared_error", verbose=1, cv=5
)  
modl.fit(X, y)
print("Best parameters:", modl.best_params_)
print("Lowest RMSE: ", (-modl.best_score_) ** (1 / 2.0))

model = modl.best_estimator_
# model = XGBRegressor(objective='reg:squarederror', n_estimators=modl.best_params_['n_estimators'], colsample_bytree=modl.best_params_['colsample_bytree'], learning_rate=modl.best_params_['learning_rate'], max_depth=modl.best_params_['max_depth'],gamma=1)
model.fit(X, y)

val = np.array(test[0, :-1]).reshape(1, -1)
pred = model.predict(val)


def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=modl.best_params_["n_estimators"],
        colsample_bytree=modl.best_params_["colsample_bytree"],
        learning_rate=modl.best_params_["learning_rate"],
        max_depth=modl.best_params_["max_depth"],
        gamma=5,
    )
    model.fit(X, y)
    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]


xgb_predict(train, test[0, :-1])


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
        pred = xgb_predict(history, X_test)
        predictions.append(pred)

        history.append(test[i])

    error = mean_squared_error(test[:, -1], predictions, squared=False)
    MAPE = mape(test[:, -1], predictions)
    return error, MAPE, test[:, -1], predictions


rmse, MAPE, y, pred = validate(snp500_data_set, 0.2)

print("RMSE: " f"{rmse}")
print("MAPE: " f"{MAPE}")
# print(y)
# print(pred)

pred = np.array(pred)
test_pred = np.c_[test, pred]
print(test_pred)

df_TP = pd.DataFrame(
    test_pred,
    columns=[
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "SMA200",
        "RSI",
        "ATR",
        "BBWidth",
        "Williams",
        "Target",
        "Pred",
    ],
)

plt.figure(figsize=(15, 9))
plt.title("Microsoft Next Day Close Price and Predicted Price", fontsize=18)
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
