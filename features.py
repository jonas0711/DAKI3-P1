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
    
    # Print kolonner med manglende værdier
    print("\nKolonner med manglende værdier:")
    missing_columns = missing_info[missing_info['Antal_manglende'] > 0]
    pd.set_option('display.max_rows', None)  # Vis alle rækker
    pd.set_option('display.max_columns', None)  # Vis alle kolonner
    pd.set_option('display.width', None)  # Undgå tekstombrydning
    print(missing_columns)
    
    # Print kolonner uden manglende værdier
    print("\nKolonner uden manglende værdier:")
    complete_columns = missing_info[missing_info['Antal_manglende'] == 0]
    print(complete_columns)
    
    # Nulstil display indstillinger
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    
    return missing_info

def select_data(dataset=None):
    '''Selecting features in the dataset'''
    if dataset is None:
        dataset = pd.read_csv(DATA_FILE)
    
    print(f"\nAntal rækker i original datasæt: {len(dataset)}")

    # Rens kolonnenavne
    dataset.columns = dataset.columns.str.replace(';', '').str.strip()
    dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

    # Fjern iso_code kolonnen hvis den findes
    if 'iso_code' in dataset.columns:
        dataset = dataset.drop('iso_code', axis=1)
        print("Fjernet iso_code fra datasættet")

    # Indlæs listen over godkendte lande
    with open('valid_countries.json', 'r') as file:
        valid_countries_data = json.load(file)
        valid_countries = valid_countries_data['valid_countries']

    # Filtrer datasættet
    original_len = len(dataset)
    dataset = dataset[dataset['country'].isin(valid_countries)]
    print(f"Antal rækker efter landefiltrering: {len(dataset)}")
    
    # Indlæs feature schema
    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    
    features = feature_schema[FEATURES_SELECTED]
    
    # Håndter missing values - 20% grænse
    features_plus_target = features + [TARGET]
    dataset_selected = dataset[features_plus_target]
    
    # Identificer kategoriske kolonner
    categorical_columns = ['country', 'continent']
    
    # Beregn procent manglende værdier for hver kolonne
    missing_pct = dataset_selected.isnull().sum() / len(dataset_selected)
    
    # Behold kun kolonner med mindre end 20% manglende værdier
    columns_to_keep = missing_pct[missing_pct < 0.2].index
    print(f"\nKolonner fjernet pga. >20% manglende værdier:")
    print(set(features_plus_target) - set(columns_to_keep))
    
    dataset_selected = dataset_selected[columns_to_keep]
    
    # Udfør one-hot encoding på kategoriske variable FØR vi fjerner NA værdier
    for col in categorical_columns:
        if col in dataset_selected.columns:
            dataset_selected = pd.get_dummies(dataset_selected, columns=[col], prefix=col)
    
    # Fjern resterende rækker med manglende værdier
    end_data = dataset_selected.dropna()
    
    print(f"\nAntal rækker før fjernelse af resterende NA: {len(dataset_selected)}")
    print(f"Antal rækker efter fjernelse af resterende NA: {len(end_data)}")

    # Gem features
    final_features = [col for col in end_data.columns if col != TARGET]
    joblib.dump(final_features, FEATURES_SELECTED)
    print(f"\nFeatures gemt som {FEATURES_SELECTED}")

    return end_data

def split_data(dataset, target=TARGET, year=YEAR_SPLIT):
    '''Splitting dataset into train and test sæt'''
    # Adskil features og target
    X = dataset.drop(columns=[target])
    y = dataset[target] 

    # Split data i trænings- og testdata baseret på årstal
    train_data = dataset[dataset['year'] <= year]
    test_data = dataset[dataset['year'] > year]

    return train_data, test_data

def scaler_data(X_train, X_test):
    '''Scaling the dataset'''
    # Normaliserer både trænings- og testdata
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_test_normalized

if __name__ == '__main__':
    select_data()