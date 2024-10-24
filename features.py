import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import json
from globale_variabler import *


def features_handling(dataset=None, year=YEAR_SPLIT):
    if dataset is None:
        dataset = pd.read_csv(DATA_FILE)

    # Erstat '\n' og mellemrum med underscore i alle kolonnenavne
    dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    
    print(feature_schema)  # Udskriv hele ordbogen
    print(FEATURES_SELECTED)
    features = feature_schema[FEATURES_SELECTED]

    target = "Value_co2_emissions_kt_by_country"

    # Fjern rækker med manglende værdier for både features og target
    end_data = dataset[features + [target]].dropna()

    # Kontroller om 'country'-kolonnen er til stede, før du udfører one-hot encoding
    if 'country' in end_data.columns:
        # Konverter kategoriske data til numeriske værdier (One-hot encoding for "country")
        end_data = pd.get_dummies(end_data, columns=["country"])
    else:
        print("Kolonnen 'country' findes ikke i datasættet.")

    # Gem de opdaterede features efter one-hot encoding (de oprindelige + one-hot encoded kolonner)
    final_features = end_data.drop(columns=[target]).columns.tolist()

    # Gem de anvendte feature-navne efter one-hot encoding
    joblib.dump(final_features, FEATURES_SELECTED)
    print(f"Features gemt som {FEATURES_SELECTED}")

    # Adskil features og target
    X = end_data.drop(columns=[target])
    y = end_data[target]

    # Split data i trænings- og testdata baseret på årstal
    train_data = end_data[end_data['year'] <= year]
    test_data = end_data[end_data['year'] > year]

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    # Normaliserer både trænings- og testdata
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Gradient Boosting Regressor model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Træner modellen
    model.fit(X_train_normalized, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test_normalized)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Gemmer den trænede model til en fil
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model gemt som {MODEL_FILENAME}")

# Kald funktionen og print resultatet
features_handling()

def select_data():
    '''Selecting features in the dataset'''

def split_data():
    '''Splitting dataset into train and test sæt'''

def scaler_data():
    '''Scaling the dataset'''

