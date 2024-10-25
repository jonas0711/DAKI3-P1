from kontrolcenter import *
import features
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def linearregression():
    '''linear regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data, "Value_co2_emissions_kt_by_country")

    X_train, X_test = features.scaler_data(X_train, X_test)

def lassoregression():
    '''Lasso regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data, "Value_co2_emissions_kt_by_country")

    X_train, X_test = features.scaler_data(X_train, X_test)

def randomforestregression():
    # Hent datasæt
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train, X_test, y_test = features.split_data(selected_data)

    # Scalering af data --> Behøver ikke at være scaleret
    #X_train = features.scaler_data()

    # Definerer modellen & træning
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Tester på træningsdataen
    y_predict_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_predict_train)

    print(f'Random forest train accuracy: {round(acc_train*100,2)}')

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
    gradientboost()
