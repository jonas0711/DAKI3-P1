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
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
import json

def print_pred_metrics(y_test, y_pred, model_type):
    '''Printing r2 and RMSE of the prediction'''
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Resultater {model_type}:")
    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

def ROC():
    '''Laver roc-curve for modellerne'''

def linearregression():
    '''linear regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    train_data, test_data = features.split_data(selected_data, TARGET)
    
    # Adskil features og target
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)
    
    # Printer r2 og rmse
    print_pred_metrics(y_test, y_pred, "Lineær regression")

def lassoregression():
    '''Lasso regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    train_data, test_data = features.split_data(selected_data, TARGET)
    
    # Adskil features og target
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = Lasso(random_state=39)
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)
    
    # Printer r2 og rmse
    print_pred_metrics(y_test, y_pred, "Lasso regression")

def ridgeregression():
    '''Rigde regression'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    train_data, test_data = features.split_data(selected_data, TARGET)
    
    # Adskil features og target
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = Ridge(random_state=39)
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)

    # Printer r2 og rmse
    print_pred_metrics(y_test, y_pred, "Rigde regression")

def randomforestregression():
    # Hent datasæt
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    train_data, test_data = features.split_data(selected_data, TARGET)

    print(train_data.head())

def randomforestregression():
    """Random forest regression model med fejlhåndtering"""
    # Hent datasæt
    print("\nIndlæser datasæt...")
    data = pd.read_csv(DATA_FILE)
    
    if len(data) == 0:
        print(f"FEJL: Ingen data fundet i {DATA_FILE}")
        return
        
    # Udvælger features
    print("Forbereder features...")
    selected_data = features.select_data(data)
    
    if len(selected_data) == 0:
        print("FEJL: Ingen data efter feature selection")
        return
        
    # Opdeling i train og test data
    print("Opdeler i trænings- og testdata...")
    train_data, test_data = features.split_data(selected_data, TARGET)
    
    if len(train_data) == 0:
        print(f"FEJL: Ingen træningsdata fundet før år {YEAR_SPLIT}")
        return
    
    if len(test_data) == 0:
        print(f"FEJL: Ingen testdata fundet efter år {YEAR_SPLIT}")
        return
    
    # Adskil features og target
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]
    
    print(f"\nAntal træningseksempler: {len(X_train)}")
    print(f"Antal testeksempler: {len(X_test)}")
    print(f"Antal features: {len(X_train.columns)}")

    # Definerer modellen & træning
    print("\nTræner Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=39)
    model.fit(X_train, y_train)
    
    # Adskil features og target
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    # Definerer modellen & træning
    model = RandomForestRegressor(n_estimators=100, random_state=39)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_

    feature_names = X_train.columns.tolist()
    
    if len(feature_names) == len(feature_importances):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance (%)': feature_importances * 100  # Konverter til procent
        })

        # Sorterer efter vigtighed (højest til lavest) og afrunder til 2 decimaler
        importance_df['Importance (%)'] = importance_df['Importance (%)'].round(2)
        importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)
        
        # Indstiller display options for at vise alle rækker
        pd.set_option('display.max_rows', None)

        print("\nFeatures importance (i procent):")
        print(importance_df)
        
        # Nulstiller display options
        pd.reset_option('display.max_rows')

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)

    # Printer r2 og rmse
    print_pred_metrics(y_test, y_pred, "Random forest regression")

def gradientboost():
    '''Gradient Boosting Regressor model'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    train_data, test_data = features.split_data(selected_data, TARGET)
    
    # Adskil features og target
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
   
    # Træner modellen
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)

    # Printer r2 og rmse
    print_pred_metrics(y_test, y_pred, "Gradient Boosting Regressor")

    # Gemmer den trænede model til en fil
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model gemt som {MODEL_FILENAME}")

def supportvector():
    '''Support Vector Regression model'''
    data = pd.read_csv(DATA_FILE)

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    train_data, test_data = features.split_data(selected_data, TARGET)
    
    # Adskil features og target
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    X_train, X_test = features.scaler_data(X_train, X_test)

    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)

    # Printer r2 og rmse
    print_pred_metrics(y_test, y_pred, "Support Vector Regression model")

if __name__ == "__main__":
    print("\n=== Random Forest Feature Importance Analysis ===")
    randomforestregression()