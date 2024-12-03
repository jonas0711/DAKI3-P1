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

    # Rens kolonnenavne for ekstra semikoloner og mellemrum
    dataset.columns = dataset.columns.str.replace(';', '').str.strip()

    # Erstat '\n' og mellemrum med underscore i alle kolonnenavne
    dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

    # Indlæs listen over godkendte lande
    with open('valid_countries.json', 'r') as file:
        valid_countries_data = json.load(file)
        valid_countries = valid_countries_data['valid_countries']

    # Filtrer datasættet til kun at indeholde godkendte lande
    dataset = dataset[dataset['country'].isin(valid_countries)]

    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    
    features = feature_schema[FEATURES_SELECTED]

    # Fjern rækker med manglende værdier for både features og target
    end_data = dataset[features + [TARGET]].dropna()

    # Liste over kategoriske kolonner der skal encoding
    categorical_columns = ['country', 'continent']
    
    # Udfør one-hot encoding på de kategoriske kolonner der findes i datasættet
    for col in categorical_columns:
        if col in end_data.columns:
            end_data = pd.get_dummies(end_data, columns=[col], prefix=col)

    # Gem de opdaterede features efter one-hot encoding
    final_features = end_data.drop(columns=[TARGET]).columns.tolist()

    # Gem de anvendte feature-navne efter one-hot encoding
    joblib.dump(final_features, FEATURES_SELECTED)
    print(f"Features gemt som {FEATURES_SELECTED}")

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