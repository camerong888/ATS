import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    make_scorer,
    roc_auc_score,
)
import time
import json

# Hyperparameters:
ticker = "^GSPC"
percentage_split = 0.2

# Choose Model Parameters:
Regression_Model = False  # False for Classification Model, True for Regression Model
gridSearch = True  # False for Random Search, True for Grid Params
num_fits = 100  # Num of Random Search attempts * 5
save_best_params = False  # True if want to save best params to a JSON file

regression_grid_params = {
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
classification_grid_params = {
    "subsample": [0.05],
    "reg_lambda": [1.75],
    "reg_alpha": [0.05],
    "n_estimators": [200],
    "min_child_weight": [13],
    "max_depth": [2],
    "learning_rate": [0.07],
    "gamma": [1.5],
    "colsample_bytree": [0.35],
}
random_params = {
    "max_depth": np.arange(2, 6, 1),
    "learning_rate": np.arange(0.01, 0.2, 0.01),
    "n_estimators": np.arange(150, 1500, 50),
    "colsample_bytree": np.arange(0.05, 1, 0.05),
    "subsample": np.arange(0.05, 1, 0.05),
    "gamma": np.arange(0, 5, 0.5),
    "min_child_weight": np.arange(1, 15, 1),
    "reg_alpha": np.arange(0, 2, 0.05),  # L1 regularization
    "reg_lambda": np.arange(0, 2, 0.05),  # L2 regularization
}

attributes_with_Categories = {
    "Open": "q",
    # "High": "q",
    # "Low": "q",
    # "Close": "q",
    # "Volume":"q",
    "Adj Close": "q",
    # "SMA10": "q",
    # "SMA20":"q",
    "SMA30": "q",
    "SMA50": "q",
    "SMA200": "q",
    # "SMA10_derivative":"q",
    # "SMA20_derivative":"q",
    "SMA30_derivative":"q",
    "SMA50_derivative":"q",
    "SMA200_derivative":"q",
    # "EMA10": "q",
    # "EMA20":"q",
    "EMA30": "q",
    "EMA50": "q",
    # "EMA10_derivative":"q",
    # "EMA20_derivative":"q",
    "EMA30_derivative":"q",
    "EMA50_derivative":"q",
    "RSI":"q",
    "ATR":"q",
    "BBWidth":"q",
    # "Williams":"q",
    "MACDs_12_26_9":"q",
    "MACD":"q",
    "VWAP": "q",
    "StochasticOscillator":"q",
    "CCI":"q",
    "OBV": "q",
    "ParabolicSAR": "q",
    "AO":"q",
    "MOM":"q",
    "BOP":"q",
    "RVI":"q",
    "DMP_16":"q",
    "DMN_16": "q",
    "MACD_12_26_9":"q",
    "MACDh_12_26_9":"q",
    "MACDs_12_26_9": "q",
    "STOCHk_14_3_3":"q",
    "STOCHd_14_3_3":"q",
    "STOCHRSIk_16_14_3_3":"q",
    "STOCHRSId_16_14_3_3":"q",
    "^VIX_Close":"q",
    "^TNX_Close": "q",
    "USO_Close": "q",
    "XLE_Close":"q",
    "SSE_Close": "q",
    # "EMASignal":"c",
    # "isPivot": "c",
    # "pattern_detected": "c",
}


# Split data
def train_test_split(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]


# Load in data
if Regression_Model:
    print("Training Regression Model")
    file_path = f"../data/indicators/{ticker}_regression_data_set_XGBOOST.csv"
else:
    print("Training Classification Model")
    file_path = f"../data/indicators/{ticker}_classification_data_set_XGBOOST.csv"
snp500_data_set = pd.read_csv(file_path)
# Split the attributes based on their type
utilized_attributes = list(attributes_with_Categories.keys()) + ["Target"]
# Select attributes for the dataframe
snp500_data_set = snp500_data_set[utilized_attributes]
# Get the feature types in the order they appear in the dataframe
feature_types = list(attributes_with_Categories.values())
# print(f"Feature types: {feature_types}")
# Convert categorical columns to 'category' data type
for attr, attr_type in attributes_with_Categories.items():
    if attr_type == "c":  # Assuming 'c' stands for categorical
        snp500_data_set[attr] = snp500_data_set[attr].astype("category")
# print(f"Data set: {snp500_data_set}")

train, test = train_test_split(snp500_data_set, percentage_split)

X = train[:, :-1]
y = train[:, -1]

startTime = time.time()
# Get best initial model
if Regression_Model:
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        seed=20,
        enable_categorical=True,
    )
    # Get best XGBOOST params
    if gridSearch:
        modl = GridSearchCV(
            estimator=xgb_model,
            param_grid=regression_grid_params,
            scoring="neg_mean_squared_error",
            verbose=1,
            cv=5,
            n_jobs=-1,  # Use all cores
        )
    else:
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
    modl.fit(X, y)
    print("Best parameters:", modl.best_params_)
    print("Lowest RMSE: ", (-modl.best_score_) ** (1 / 2.0))
    endTime = time.time()
    print(f"Parameters Fit Time: {(endTime-startTime)/60}")
else:
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        seed=20,
        enable_categorical=True,
    )
    # Get best XGBOOST params
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    if gridSearch:
        modl = GridSearchCV(
            estimator=xgb_model,
            param_grid=classification_grid_params,
            scoring=auc_scorer,
            verbose=1,
            cv=5,
            n_jobs=-1,  # Use all cores
        )
    else:
        modl = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=random_params,
            n_iter=num_fits,  # Number of parameter settings sampled
            scoring=auc_scorer,
            verbose=1,
            cv=5,
            n_jobs=-1,  # Use all cores
            random_state=20,
        )
    modl.fit(X, y)
    print("Best parameters:", modl.best_params_)
    print("Best AUC: ", modl.best_score_)
endTime = time.time()
print(f"Parameter Fit Time: {(endTime-startTime)/60}")
model = modl.best_estimator_

if save_best_params:
    best_params = modl.best_params_
    # Save best params to a JSON file
    with open(f"../models/{ticker}_XGBOOST_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)


def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    if Regression_Model:
        model = XGBRegressor(
            objective="reg:squarederror",
            enable_categorical=True,
            subsample=modl.best_params_["subsample"],
            reg_lambda=modl.best_params_["reg_lambda"],
            reg_alpha=modl.best_params_["reg_alpha"],
            n_estimators=modl.best_params_["n_estimators"],
            min_child_weight=modl.best_params_["min_child_weight"],
            max_depth=modl.best_params_["max_depth"],
            learning_rate=modl.best_params_["learning_rate"],
            gamma=modl.best_params_["gamma"],
            colsample_bytree=modl.best_params_["colsample_bytree"],
        )
    else:
        model = XGBClassifier(
            objective="binary:logistic",
            enable_categorical=True,
            subsample=modl.best_params_["subsample"],
            reg_lambda=modl.best_params_["reg_lambda"],
            reg_alpha=modl.best_params_["reg_alpha"],
            n_estimators=modl.best_params_["n_estimators"],
            min_child_weight=modl.best_params_["min_child_weight"],
            max_depth=modl.best_params_["max_depth"],
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

    if Regression_Model:
        error = mean_squared_error(test[:, -1], predictions, squared=False)
        MAPE = mape(test[:, -1], predictions)
        return error, MAPE, test[:, -1], predictions, test
    else:
        accuracy = accuracy_score(test[:, -1], predictions)
        auc_score = roc_auc_score(test[:, -1], predictions)
        return accuracy, auc_score, test[:, -1], predictions, test


startTime = time.time()
if Regression_Model:
    rmse, MAPE, y, pred, test = validate(snp500_data_set, percentage_split)
    print("RMSE: " f"{rmse}")
    print("MAPE: " f"{MAPE}")
else:
    accuracy, auc_score, y, pred, test = validate(snp500_data_set, percentage_split)
    print("Accuracy:", accuracy)
    print("AUC:", auc_score)    
endTime = time.time()
print(f"Prediction time: {(endTime-startTime)/60}")

pred = np.array(pred)

test_pred = np.c_[test, pred]
plotting_attributes = utilized_attributes + ["Pred"]
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
if Regression_Model:
    snp500_data_set.to_csv(
        f"../results/{ticker}_regression_predictions.csv", index=False
    )
else:
    snp500_data_set.to_csv(
        f"../results/{ticker}_classification_predictions.csv", index=False
    )
