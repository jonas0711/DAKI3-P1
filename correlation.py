from kontrolcenter import *
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Indlæser og forbereder data til korrelationsanalyse.
    Inkluderer kun godkendte lande fra valid_countries.json
    """
    dataset = pd.read_csv(DATA_FILE)
    
    # Indlæs godkendte lande
    with open('valid_countries.json', 'r') as file:
        valid_countries_data = json.load(file)
        valid_countries = valid_countries_data['valid_countries']
    
    # Filtrer til godkendte lande
    dataset = dataset[dataset['country'].isin(valid_countries)]
    
    # Rens kolonnenavne
    dataset.columns = dataset.columns.str.replace(';', '').str.strip()
    dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')
    
    return dataset

def get_selected_features():
    """
    Henter de valgte features fra konfigurationsfilen
    """
    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    
    features_list = feature_schema[FEATURES_SELECTED]
    
    # Fjern 'country' og 'year' hvis de findes
    features_to_remove = ['country', 'year', 'iso_code']
    for feature in features_to_remove:
        if feature in features_list:
            features_list.remove(feature)
    
    return features_list


def plot_correlation_heatmap(correlation_matrix, title="Korrelationsmatrix"):
    """
    Plotter en korrelationsmatrix som et heatmap
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,  # Vis værdier i cellerne
                cmap='coolwarm',  # Farvepalette
                center=0,  # Centrer farveskalaen omkring 0
                fmt='.2f',  # Vis 2 decimaler
                square=True)  # Gør cellerne kvadratiske
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Gem plottet
    plt.savefig('Billeder/correlation_heatmap.png')
    plt.close()

def analyze_correlations():
    """
    Hovedfunktion der udfører korrelationsanalyse og printer resultater
    """
    # Indlæs og forbered data
    dataset = load_and_prepare_data()
    
    # Hent features
    features_list = get_selected_features()
    
    # Tilføj target variabel og fjern ikke-numeriske kolonner
    analysis_features = features_list + [TARGET]
    final_dataset = dataset[analysis_features].select_dtypes(include=['float64', 'int64'])
    
    # Beregn korrelationer
    correlation_matrix = final_dataset.corr()
    
    print("Korrelationer med CO2-udledning:")
    correlations_with_target = correlation_matrix[TARGET].sort_values(ascending=False)
    print(correlations_with_target)
    
    # Plot korrelationsmatrix
    plot_correlation_heatmap(correlation_matrix)
    
    return final_dataset, correlation_matrix

if __name__ == "__main__":
    analyze_correlations()