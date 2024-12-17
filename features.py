import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import json
from kontrolcenter import *

def analyze_missing_data(dataset):
    '''Analyserer manglende data i datasættet'''
    # Beregn antal manglende værdier for hver kolonne
    missing_values = dataset.isnull().sum()
    # Beregn procentdel af manglende værdier og afrund til 2 decimaler
    missing_percent = (missing_values / len(dataset) * 100).round(2)
    
    # Opret en DataFrame med statistik om manglende data
    missing_info = pd.DataFrame({
        'Antal_manglende': missing_values,
        'Procent_manglende': missing_percent,
        'Antal_unikke': dataset.nunique(),
        'Datatype': dataset.dtypes
    })
    
    # Sortér efter procent manglende værdier (højeste først)
    missing_info = missing_info.sort_values('Procent_manglende', ascending=False)
    
    return missing_info

def select_data(dataset=None):
    '''Vælger og forbereder data til analyse'''
    # Hvis intet datasæt er angivet, indlæs fra standardfil
    if dataset is None:
        dataset = pd.read_csv(DATA_FILE)

    # Rens kolonnenavne for specialtegn og mellemrum
    dataset.columns = dataset.columns.str.replace(';', '').str.strip()
    dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

    # Fjern iso_code kolonnen hvis den eksisterer
    if 'iso_code' in dataset.columns:
        dataset = dataset.drop('iso_code', axis=1) #axis=1 betyder kolonne

    # Indlæs liste over gyldige lande fra JSON-fil
    with open('valid_countries.json', 'r') as file:
        valid_countries_data = json.load(file)
        valid_countries = valid_countries_data['valid_countries']

    # Filtrer datasættet til kun at indeholde gyldige lande
    dataset = dataset[dataset['country'].isin(valid_countries)]
    
    # Indlæs feature konfiguration fra JSON-fil
    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    
    # Hent liste over valgte features
    features = feature_schema[FEATURES_SELECTED]
    
    # Tilføj obligatoriske kolonner hvis de ikke allerede er inkluderet
    required_columns = ['year', 'country']
    for col in required_columns:
        if col not in features and col in dataset.columns:
            features.append(col)
    
    # Tilføj målvariabel til feature listen
    features_plus_target = features + [TARGET]
    
    # Vælg kun de specificerede kolonner
    dataset_selected = dataset[features_plus_target]

    # Fjern rækker med manglende værdier
    end_data = dataset_selected.dropna()

    return end_data

def split_data(dataset, target=TARGET, year=YEAR_SPLIT):
    '''Opdeler datasættet i trænings- og testsæt'''
    
    # Opdel data i trænings- og testdata baseret på år
    train_data = dataset[dataset['year'].isin(TRAIN_YEARS)].copy()
    test_data = dataset[dataset['year'].isin(TEST_YEARS)].copy()
    
    return train_data, test_data

def scaler_data(X_train, X_test):
    '''Skalerer datasættet'''
    # Initialiser StandardScaler objektet
    scaler = StandardScaler()
    # Tilpas og transformér træningsdata
    X_train_normalized = scaler.fit_transform(X_train)
    # Transformér testdata med samme skaleringsfaktorer
    X_test_normalized = scaler.transform(X_test)
    
    return X_train_normalized, X_test_normalized

# Kør kun select_data hvis scriptet køres direkte
if __name__ == '__main__':
    select_data()