import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import json
from kontrolcenter import *

def analyze_missing_data(dataset):
    '''Analyserer manglende data i datasættet'''
    # Beregn antal manglende værdier og procent for hver kolonne
    missing_values = dataset.isnull().sum()
    missing_percent = (missing_values / len(dataset) * 100).round(2)
    
    # Lav en DataFrame med resultaterne
    missing_info = pd.DataFrame({
        'Antal_manglende': missing_values,
        'Procent_manglende': missing_percent,
        'Antal_unikke': dataset.nunique(),
        'Datatype': dataset.dtypes
    })
    
    # Sorter efter procent manglende værdier (højeste først)
    missing_info = missing_info.sort_values('Procent_manglende', ascending=False)
    
    return missing_info

def select_data(dataset=None):
    '''Selecting features in the dataset'''
    if dataset is None:
        dataset = pd.read_csv(DATA_FILE)
    
    print(f"\nAntal rækker i original datasæt: {len(dataset)}")

    # Rens kolonnenavne
    dataset.columns = dataset.columns.str.replace(';', '').str.strip()
    dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

    # Sikr at year kolonnen er numerisk
    if 'year' in dataset.columns:
        dataset['year'] = pd.to_numeric(dataset['year'], errors='coerce')

    # Fjern iso_code kolonnen hvis den findes
    if 'iso_code' in dataset.columns:
        dataset = dataset.drop('iso_code', axis=1)
        print("Fjernet iso_code fra datasættet")

    # Indlæs listen over godkendte lande
    with open('valid_countries.json', 'r') as file:
        valid_countries_data = json.load(file)
        valid_countries = valid_countries_data['valid_countries']

    # Filtrer datasættet
    dataset = dataset[dataset['country'].isin(valid_countries)]
    print(f"Antal rækker efter landefiltrering: {len(dataset)}")
    
    # Indlæs feature schema
    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    
    features = feature_schema[FEATURES_SELECTED]
    
    # Sikr at nødvendige kolonner er inkluderet
    required_columns = ['year', 'country']
    for col in required_columns:
        if col not in features and col in dataset.columns:
            features.append(col)
    
    # Tilføj target variabel
    features_plus_target = features + [TARGET]
    
    # Vælg kun relevante år
    valid_years = TRAIN_YEARS + TEST_YEARS
    dataset = dataset[dataset['year'].isin(valid_years)]
    
    dataset_selected = dataset[features_plus_target]
    
    # Håndter missing values
    print("\nKolonner fjernet pga. >20% manglende værdier:")
    missing_pct = dataset_selected.isnull().sum() / len(dataset_selected)
    columns_to_keep = missing_pct[missing_pct < 0.2].index
    print(set(features_plus_target) - set(columns_to_keep))
    
    dataset_selected = dataset_selected[columns_to_keep]
    
    # Fjern rækker med manglende værdier
    end_data = dataset_selected.dropna()
    
    print(f"\nAntal rækker før fjernelse af resterende NA: {len(dataset_selected)}")
    print(f"Antal rækker efter fjernelse af resterende NA: {len(end_data)}")

    return end_data

def split_data(dataset, target=TARGET, year=YEAR_SPLIT):
    '''Splitting dataset into train and test sæt'''
    if 'year' not in dataset.columns:
        raise ValueError("Kolonnen 'year' mangler i datasættet. Den er nødvendig for at opdele data.")
    
    # Split data i trænings- og testdata baseret på år
    train_data = dataset[dataset['year'].isin(TRAIN_YEARS)].copy()
    test_data = dataset[dataset['year'].isin(TEST_YEARS)].copy()
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Advarsel: Ingen data i enten trænings- eller testsættet!")
        print(f"Træningsdata: {len(train_data)} rækker")
        print(f"Testdata: {len(test_data)} rækker")
    
    return train_data, test_data

def scaler_data(X_train, X_test):
    '''Scaling the dataset'''
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    return X_train_normalized, X_test_normalized

if __name__ == '__main__':
    select_data()