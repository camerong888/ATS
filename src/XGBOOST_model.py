import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from xgboost import XGBRegressor, XGBClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    make_scorer,
    roc_auc_score,
)
import time
import json
from tqdm import tqdm

# Hyperparameters:
ticker = "^GSPC"
percentage_split = 0.2

# Choose Model Parameters:
Regression_Model = False  # False for Classification Model, True for Regression Model
gridSearch = True  # False for Random Search, True for Grid Params
num_fits = 5000  # Num of Random Search attempts * 5
folds = 5 # CV
save_results = True  # True if want to save best params to a JSON file

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
    "subsample": [0.6499999999999999],  # [0.2],
    "reg_lambda": [0.4],
    "reg_alpha": [0.0],
    "n_estimators": [50],
    "min_child_weight": [2],
    "max_depth": [6],
    "learning_rate": [0.3900000000000002],
    "gamma": [0.5],
    "colsample_bytree": [0.7],
}
random_params = {
    "max_depth": np.arange(2, 7, 1),
    "learning_rate": np.arange(0.25, 0.45, 0.01),
    "n_estimators": np.arange(25, 400, 50),
    "colsample_bytree": np.arange(0.4, 0.9, 0.05),
    "subsample": np.arange(0.4, 0.9, 0.05),
    "gamma": np.arange(0, 2, 0.2),
    "min_child_weight": np.arange(2, 7, 1),
    "reg_alpha": np.arange(0, 1, 0.1),  # L1 regularization
    "reg_lambda": np.arange(0, 1, 0.1),  # L2 regularization
}

attributes_with_Categories = {
    # "Open": "q",
    # "High": "q", #
    # "Low": "q", #
    # "Close": "q", #
    # "Volume":"q", #
    "Adj Close": "q",
    "High - Low": "q",
    # "SMA10": "q", #
    "SMA20": "q",  #
    "SMA30": "q",  #
    "SMA50": "q",  #
    "SMA200": "q",  #
    # "SMA10_derivative":"q", #
    "SMA20_derivative": "q",
    "SMA30_derivative": "q",  #
    "SMA50_derivative": "q",
    "SMA200_derivative": "q",
    # "EMA10": "q", #
    "EMA20": "q",  #
    "EMA30": "q",  #
    "EMA50": "q",  #
    "EMA10_derivative": "q",  #
    "EMA20_derivative": "q",  #
    "EMA30_derivative": "q",
    "EMA50_derivative": "q",
    "RSI": "q",
    "ATR": "q",
    "BBWidth": "q",
    # "Williams":"q", #
    "MACDs_12_26_9": "q",
    "MACD": "q",
    "VWAP": "q",  #
    "StochasticOscillator": "q",
    "CCI": "q",
    "OBV": "q",
    "ParabolicSAR": "q",
    "AO": "q",
    "MOM": "q",
    "BOP": "q",
    "RVI": "q",
    "DMP_16": "q",
    "DMN_16": "q",
    "MACD_12_26_9": "q",
    "MACDh_12_26_9": "q",
    "MACDs_12_26_9": "q",
    "STOCHk_14_3_3": "q",
    "STOCHd_14_3_3": "q",
    "STOCHRSIk_16_14_3_3": "q",
    "STOCHRSId_16_14_3_3": "q",
    "^VIX_Close": "q",
    "^TNX_Close": "q",
    "USO_Close": "q",
    "XLE_Close": "q",
    "SSE_Close": "q",
    # "EMASignal":"c", #
    "isPivot": "c",
    # "pattern_detected": "c", #
    "CDL_2CROWS": "c",
    "CDL_3BLACKCROWS": "c",
    "CDL_3INSIDE": "c",
    "CDL_3LINESTRIKE": "c",
    "CDL_3OUTSIDE": "c",
    "CDL_3STARSINSOUTH": "c",
    "CDL_3WHITESOLDIERS": "c",
    "CDL_ABANDONEDBABY": "c",
    "CDL_ADVANCEBLOCK": "c",
    "CDL_BELTHOLD": "c",
    "CDL_BREAKAWAY": "c",
    "CDL_CLOSINGMARUBOZU": "c",
    "CDL_CONCEALBABYSWALL": "c",
    "CDL_COUNTERATTACK": "c",
    "CDL_DARKCLOUDCOVER": "c",
    "CDL_DOJI_10_0.1": "c",
    "CDL_DOJISTAR": "c",
    "CDL_DRAGONFLYDOJI": "c",
    "CDL_ENGULFING": "c",
    "CDL_EVENINGDOJISTAR": "c",
    "CDL_EVENINGSTAR": "c",
    "CDL_GAPSIDESIDEWHITE": "c",
    "CDL_GRAVESTONEDOJI": "c",
    "CDL_HAMMER": "c",
    "CDL_HANGINGMAN": "c",
    "CDL_HARAMI": "c",
    "CDL_HARAMICROSS": "c",
    "CDL_HIGHWAVE": "c",
    "CDL_HIKKAKE": "c",
    "CDL_HIKKAKEMOD": "c",
    "CDL_HOMINGPIGEON": "c",
    "CDL_IDENTICAL3CROWS": "c",
    "CDL_INNECK": "c",
    "CDL_INSIDE": "c",
    "CDL_INVERTEDHAMMER": "c",
    "CDL_KICKING": "c",
    "CDL_KICKINGBYLENGTH": "c",
    "CDL_LADDERBOTTOM": "c",
    "CDL_LONGLEGGEDDOJI": "c",
    "CDL_LONGLINE": "c",
    "CDL_MARUBOZU": "c",
    "CDL_MATCHINGLOW": "c",
    "CDL_MATHOLD": "c",
    "CDL_MORNINGDOJISTAR": "c",
    "CDL_MORNINGSTAR": "c",
    "CDL_ONNECK": "c",
    "CDL_PIERCING": "c",
    "CDL_RICKSHAWMAN": "c",
    "CDL_RISEFALL3METHODS": "c",
    "CDL_SEPARATINGLINES": "c",
    "CDL_SHOOTINGSTAR": "c",
    "CDL_SHORTLINE": "c",
    "CDL_SPINNINGTOP": "c",
    "CDL_STALLEDPATTERN": "c",
    "CDL_STICKSANDWICH": "c",
    "CDL_TAKURI": "c",
    "CDL_TASUKIGAP": "c",
    "CDL_THRUSTING": "c",
    "CDL_TRISTAR": "c",
    "CDL_UNIQUE3RIVER": "c",
    "CDL_UPSIDEGAP2CROWS": "c",
    "CDL_XSIDEGAP3METHODS": "c",
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
    if attr_type == "c":  
        snp500_data_set[attr] = snp500_data_set[attr].astype("category")

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
            verbose=10,
            cv=folds,
            n_jobs=-1,  # Use all cores
        )
    else:
        modl = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=random_params,
            n_iter=num_fits,  # Number of parameter settings sampled
            scoring="neg_mean_squared_error",
            verbose=10,
            cv=folds,
            n_jobs=-1,  # Use all cores
            random_state=int(time.time()),
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
            verbose=10,
            cv=folds,
            n_jobs=-1,  # Use all cores
        )
    else:
        modl = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=random_params,
            n_iter=num_fits,  # Number of parameter settings sampled
            scoring=auc_scorer,
            verbose=10,
            cv=folds,
            n_jobs=-1,  # Use all cores
            random_state=int(time.time()),
        )
    modl.fit(X, y, verbose=True)
    print("Best parameters:", modl.best_params_)
    print("Best AUC: ", modl.best_score_)
endTime = time.time()
print(f"Parameter Fit Time: {(endTime-startTime)/60}")
model = modl.best_estimator_


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

    for i in tqdm(range(len(test)), desc="Validation"):
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
plt.title(f"{ticker} XGBOOST Classification Price Prediction", fontsize=18)
plt.plot(df_TP["Adj Close"], label="Stock Price", color="blue", alpha=0.5)
plt.scatter(df_TP.index, df_TP["Adj Close"].where((df_TP["Target"] == 1.0) & (df_TP["Pred"] == 1)), color='green', label='Actual and Predicted Up', marker='^')
plt.scatter(df_TP.index, df_TP["Adj Close"].where((df_TP["Target"] == 0.0) & (df_TP["Pred"] == 0)), color='lime', label='Actual Down and Predicted Down', marker='v')
plt.scatter(df_TP.index, df_TP["Adj Close"].where((df_TP["Target"] == 1.0) & (df_TP["Pred"] == 0)), color='red', label='Actual Up, Predicted Down', marker='^', alpha=0.5)
plt.scatter(df_TP.index, df_TP["Adj Close"].where((df_TP["Target"] == 0.0) & (df_TP["Pred"] == 1)), color='darkred', label='Actual Down, Predicted Up', marker='v', alpha=0.5)
plt.xlabel("Day", fontsize=18)
plt.legend(loc="upper left")
plt.ylabel("Stock Price ($)", fontsize=18)
plt.tight_layout()
plt.show()


# After fitting the model
feature_importances = model.feature_importances_
importance_series = pd.Series(feature_importances, index=snp500_data_set.columns[:-1])
importance_series_sorted = importance_series.sort_values(ascending=False)
plt.figure(figsize=(15, 9))
importance_series_sorted.plot(kind="bar", title="Feature Importances")
plt.ylabel('Importance', fontsize=18)
plt.xlabel('Features', fontsize=18)
# plt.xticks(rotation=45, ha='right') # Rotate the feature names for better readability
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.tight_layout() # Ensure the labels don't get cut off
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

if save_results:
    best_params = {k: int(v) if isinstance(v, np.integer) else v for k, v in modl.best_params_.items()}
    initial_AUC = float(modl.best_score_)
    final_AUC = float(auc_score)
    final_accuracy = float(accuracy)
    feature_importances_dict = importance_series_sorted.to_dict()
    results_data = {
        "Best Parameters": best_params,
        "Initial AUC": initial_AUC,
        "Final AUC": final_AUC,
        "Final Accuracy": final_accuracy,
        "Feature Importances": feature_importances_dict,
        "Utilized Attributes": utilized_attributes
    }
    filename = f"../results/logs/{ticker}_XGBClassifier_result_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(results_data, f, indent=4)
        print(f"Results saved to {filename}")

# plot single tree
plot_tree(model)
plt.gcf().set_size_inches(150, 100)
plt.show()

