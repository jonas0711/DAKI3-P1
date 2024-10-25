from kontrolcenter import *
import features
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import json

def linearregression():
    '''linear regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data, "Value_co2_emissions_kt_by_country")

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Lineær regression")
    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")


def lassoregression():
    '''Lasso regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data, "Value_co2_emissions_kt_by_country")

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = Lasso()
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Lasso regression")
    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

def rigderegression():
    '''Lasso regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data, "Value_co2_emissions_kt_by_country")

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = Ridge()
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Rigde regression")
    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

def randomforestregression():
    # Hent datasæt
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data, "Value_co2_emissions_kt_by_country")

    # Definerer modellen & træning
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_

    feature_names = X_train.columns.tolist()
    
    if len(feature_names) == len(feature_importances):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })

        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        print("\nFeatures importance:")
        print(importance_df)

    # Udskriv resultatet
    print("\nFeatures importance:")
    print(importance_df)
    # Forudsigelser på testdata
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Random forest regression")
    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

def gradientboost():
    '''Gradient Boosting Regressor model'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data, "Value_co2_emissions_kt_by_country")

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
   
    # Træner modellen
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Gemmer den trænede model til en fil
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model gemt som {MODEL_FILENAME}")

if __name__ == "__main__":
    rigderegression()
