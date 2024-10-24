import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import json
from globale_variabler import *


def select_data(dataset):
    '''Selecting features in the dataset'''
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

    return end_data

def split_data(dataset, target, year=YEAR_SPLIT):
    '''Splitting dataset into train and test sæt'''
    # Adskil features og target
    X = dataset.drop(columns=[target])
    y = dataset[target]

    # Split data i trænings- og testdata baseret på årstal
    train_data = dataset[dataset['year'] <= year]
    test_data = dataset[dataset['year'] > year]

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]
    return X_train, y_train, X_test, y_test

def scaler_data(X_train, X_test):
    '''Scaling the dataset'''
    # Normaliserer både trænings- og testdata
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_test_normalized

