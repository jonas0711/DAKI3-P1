from kontrolcenter import *
import features
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import joblib
import json

def get_cluster_countries():
    """Henter liste over lande for den valgte cluster"""
    with open(CLUSTERS_FILE, 'r') as f:
        clusters = json.load(f)
    return clusters[SELECTED_CLUSTER]['countries']

def filter_data_by_cluster(data):
    """Filtrerer datasættet baseret på den valgte cluster"""
    cluster_countries = get_cluster_countries()
    return data[data['country'].isin(cluster_countries)]

def print_pred_metrics(y_test, y_pred, model_type, metrics_type="all"):
    '''Printer valgte evalueringsmetrikker inklusiv procentvis afvigelse'''
    if metrics_type in ["r2_mse", "all"]:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"\nResultater for {model_type}:")
        print(f"R² score: {r2:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
    
    if metrics_type in ["rmse", "all"]:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # Beregn gennemsnit af faktiske værdier til procentvis afvigelse
        mean_actual = np.mean(np.abs(y_test))
        percentage_error = (rmse / mean_actual) * 100
        
        print(f"\nResultater for {model_type}:")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Procentvis afvigelse: {percentage_error:.2f}%")
        print(f"(baseret på gennemsnitlig absolut værdi: {mean_actual:.4f})")

def prepare_data():
    """Forbereder data til modeltræning"""
    print("\nIndlæser datasæt...")
    data = pd.read_csv(DATA_FILE)
    
    print(f"\nFiltrerer data for cluster: {SELECTED_CLUSTER}")
    data = filter_data_by_cluster(data)

    print("Forbereder features...")
    selected_data = features.select_data(data)

    if 'year' not in selected_data.columns:
        raise ValueError("Fejl: 'year' kolonnen mangler i datasættet")

    print("Opdeler i trænings- og testdata...")
    train_data, test_data = features.split_data(selected_data, TARGET)

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Fejl: Ikke nok data til at træne/teste modellen")

    # Adskil features og target, og fjern 'year' og 'country' kolonner
    feature_cols = [col for col in train_data.columns if col not in [TARGET, 'year', 'country']]
    X_train = train_data[feature_cols]
    y_train = train_data[TARGET]
    X_test = test_data[feature_cols]
    y_test = test_data[TARGET]

    print(f"\nAntal træningseksempler: {len(X_train)}")
    print(f"Antal testeksempler: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def randomforestregression():
    """Random Forest Regression model med R² og MSE evaluering"""
    try:
        X_train, X_test, y_train, y_test = prepare_data()
        
        print("\nTræner Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=39,
            n_jobs=-1  # Bruger alle tilgængelige CPU-kerner
        )
        model.fit(X_train, y_train)
        
        # Forudsigelser på testdata
        y_pred = model.predict(X_test)
        
        # Evaluering
        print_pred_metrics(y_test, y_pred, "Random Forest Regression", "r2_mse")
        
        # Gem modellen
        model_filename = f'random_forest_model_{SELECTED_CLUSTER}.joblib'
        joblib.dump(model, model_filename)
        print(f"\nModel gemt som: {model_filename}")
        
    except Exception as e:
        print(f"\nFejl i Random Forest træning: {str(e)}")

def gradientboost():
    '''Gradient Boosting Regressor model med fuld evaluering'''
    try:
        X_train, X_test, y_train, y_test = prepare_data()
        
        print("\nTræner Gradient Boosting model...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=39,
            verbose=1  # Vis træningsfremskridt
        )
        model.fit(X_train, y_train)
        
        # Forudsigelser på testdata
        y_pred = model.predict(X_test)
        
        # Evaluering - nu med alle metrics
        print_pred_metrics(y_test, y_pred, "Gradient Boosting Regression", "all")
        
        # Gem modellen
        model_filename = f'gradient_boosting_model_{SELECTED_CLUSTER}.joblib'
        joblib.dump(model, model_filename)
        print(f"\nModel gemt som: {model_filename}")
        
    except Exception as e:
        print(f"\nFejl i Gradient Boosting træning: {str(e)}")

def supportvector():
    '''Support Vector Regression model med R² og MSE evaluering'''
    try:
        X_train, X_test, y_train, y_test = prepare_data()
        
        print("\nTræner Support Vector Regression model...")
        model = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            verbose=True
        )
        model.fit(X_train, y_train)
        
        # Forudsigelser på testdata
        y_pred = model.predict(X_test)
        
        # Evaluering
        print_pred_metrics(y_test, y_pred, "Support Vector Regression", "r2_mse")
        
        # Gem modellen
        model_filename = f'svr_model_{SELECTED_CLUSTER}.joblib'
        joblib.dump(model, model_filename)
        print(f"\nModel gemt som: {model_filename}")
        
    except Exception as e:
        print(f"\nFejl i Support Vector Regression træning: {str(e)}")

def print_model_comparison(models_results):
    """Printer en sammenligning af modelresultater"""
    print("\n=== Model Sammenligning ===")
    print("Model                  R²      RMSE")
    print("-" * 40)
    for model_name, metrics in models_results.items():
        print(f"{model_name:<20} {metrics['r2']:.4f}  {metrics['rmse']:.4f}")

if __name__ == "__main__":
    print(f"\n=== Træner og evaluerer modeller for {SELECTED_CLUSTER} ===")
    
    try:
        randomforestregression()
        gradientboost()
        supportvector()
        
    except Exception as e:
        print(f"\nDer opstod en fejl under modelkørslen: {str(e)}")