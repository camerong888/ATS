import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import time
import json

# Hyperparameters:
ticker = "^GSPC"
percentage_split = 0.2

gridSearch = False
num_fits = 2000

grid_params = {
    "subsample": [0.4],
    "reg_lambda": [0.4],
    "reg_alpha": [0.7],
    "n_estimators": [400], 
    "min_child_weight": [10],
    "max_depth": [2],
    "learning_rate": [0.06], 
    "gamma": [3.5],
    "colsample_bytree": [0.35], 
}
random_params = {
    "max_depth": np.arange(2, 4, 1),
    "learning_rate": np.arange(0.01, 0.15, 0.01),
    "n_estimators": np.arange(200, 1500, 50),
    "colsample_bytree": np.arange(0.1, 1, 0.05),
    "subsample": np.arange(0.1, 1, 0.05),
    "gamma": np.arange(0, 5, 0.5),
    "min_child_weight": np.arange(1, 15, 1),
    "reg_alpha": np.arange(0.05, 1, 0.05),  # L1 regularization
    "reg_lambda": np.arange(0.05, 2, 0.05),  # L2 regularization
}

attributes = [
    "Open",
    "High",
    "Low",
    "Close",
    # "Volume",
    "Adj Close",
    "SMA10",
    # "SMA20",
    "SMA30",
    "SMA50",
    "SMA200",
    # "SMA10_derivative",
    # "SMA20_derivative",
    # "SMA30_derivative",
    # "SMA50_derivative",
    # "SMA200_derivative",
    "EMA10",
    # "EMA20",
    "EMA30",
    "EMA50",
    # "EMA10_derivative",
    # "EMA20_derivative",
    # "EMA30_derivative",
    # "EMA50_derivative",
    # "RSI",
    # "ATR",
    # "BBWidth",
    # "Williams",
    # "MACDs_12_26_9",
    # "MACD",
    "VWAP",
    # "StochasticOscillator",
    # "CCI",
    "OBV",
    "ParabolicSAR",
    # "AO",
    # "MOM",
    # "BOP",
    # "RVI",
    # "DMP_16",
    "DMN_16",
    # "MACD_12_26_9",
    # "MACDh_12_26_9",
    "MACDs_12_26_9",
    # "STOCHk_14_3_3",
    # "STOCHd_14_3_3",
    # "STOCHRSIk_16_14_3_3",
    # "STOCHRSId_16_14_3_3",
    # "^VIX_Close",
    "^TNX_Close",
    "USO_Close",
    # "XLE_Close",
    "SSE_Close",
    # "EMASignal",
    # "isPivot",
    # "pattern_detected",
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

# Get best XGBOOST params
if gridSearch:
    startTime = time.time()
    xgb_model = XGBRegressor(objective="reg:squarederror", seed=20)
    modl = GridSearchCV(
        estimator=xgb_model,
        param_grid=grid_params,
        scoring="neg_mean_squared_error",
        verbose=1,
        cv=5,
        n_jobs=-1,  # Use all cores
    )
    modl.fit(X, y)
    print("Best parameters:", modl.best_params_)
    print("Lowest RMSE: ", (-modl.best_score_) ** (1 / 2.0))
    endTime = time.time()
    print(f"Parameters Fit Time: {(endTime-startTime)/60}")
    
else:
    # Get optimal parameters:
    startTime = time.time()
    xgb_model = XGBRegressor(objective="reg:squarederror", seed=20)
    modl = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=random_params,
        n_iter=num_fits,  # Number of parameter settings sampled
        scoring="neg_mean_squared_error",
        verbose=1,
        cv=5,
        n_jobs=-1,  # Use all cores
        random_state=20,
    )
    # Fit the random search model
    modl.fit(X, y)
    print("Best parameters:", modl.best_params_)
    print("Lowest RMSE: ", (-modl.best_score_) ** (1 / 2.0))
    endTime = time.time()
    print(f"Parameters Fit Time: {(endTime-startTime)/60}")

model = modl.best_estimator_


best_params = modl.best_params_
# Save best params to a JSON file
with open(f'../models/{ticker}_XGBOOST_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    model = XGBRegressor(
        objective="reg:squarederror",
        subsample = modl.best_params_["subsample"],
        reg_lambda = modl.best_params_["reg_lambda"],
        reg_alpha = modl.best_params_["reg_alpha"],
        n_estimators=modl.best_params_["n_estimators"],
        min_child_weight = modl.best_params_["min_child_weight"],
        max_depth = modl.best_params_["max_depth"],
        learning_rate=modl.best_params_["learning_rate"],
        gamma=modl.best_params_["gamma"],
        colsample_bytree=modl.best_params_["colsample_bytree"],
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
        # print(f"Attributes: {X_test}")
        # print(f"Target: {y_test}")
        pred = xgb_predict(history, X_test)
        # print(f"Prediction: {pred}")
        predictions.append(pred)
        history.append(test[i])

    error = mean_squared_error(test[:, -1], predictions, squared=False)
    MAPE = mape(test[:, -1], predictions)
    return error, MAPE, test[:, -1], predictions, test


startTime = time.time()
rmse, MAPE, y, pred, test = validate(snp500_data_set, percentage_split)
print("RMSE: " f"{rmse}")
print("MAPE: " f"{MAPE}")
endTime = time.time()
print(f"Prediction time: {(endTime-startTime)/60}")

pred = np.array(pred)

test_pred = np.c_[test, pred]
plotting_attributes = attributes + ["Pred"]
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
snp500_data_set["Predicted"] = pred
snp500_data_set.to_csv(f"../results/{ticker}_predictions.csv", index=False)
