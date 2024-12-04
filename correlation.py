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
    print(f"\nIndlæser data fra: {DATA_FILE}")
    dataset = pd.read_csv(DATA_FILE)
    
    # Indlæs godkendte lande
    with open('valid_countries.json', 'r') as file:
        valid_countries_data = json.load(file)
        valid_countries = valid_countries_data['valid_countries']
    
    # Filtrer til godkendte lande
    initial_rows = len(dataset)
    dataset = dataset[dataset['country'].isin(valid_countries)]
    filtered_rows = len(dataset)
    
    print(f"\nAntal rækker før landefiltrering: {initial_rows}")
    print(f"Antal rækker efter landefiltrering: {filtered_rows}")
    print(f"Antal fjernede rækker: {initial_rows - filtered_rows}")
    
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
    features_to_remove = ['country', 'year', 'iso_code', 'continent']
    for feature in features_to_remove:
        if feature in features_list:
            features_list.remove(feature)
    
    return features_list

def handle_missing_data(dataset, threshold=0.20):
    """
    Håndterer manglende data med 20% grænseværdi
    """
    # Beregn procent manglende værdier
    missing_percentages = dataset.isnull().sum() / len(dataset)
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    
    if len(columns_to_drop) > 0:
        print(f"\nFølgende kolonner droppes pga. for mange manglende værdier (>{threshold*100}%):")
        for col in columns_to_drop:
            print(f"- {col}: {missing_percentages[col]*100:.1f}% manglende")
    
    # Fjern kolonner og rækker med manglende data
    clean_dataset = dataset.drop(columns=columns_to_drop)
    final_dataset = clean_dataset.dropna()
    
    print(f"\nAntal rækker før cleaning: {len(dataset)}")
    print(f"Antal rækker efter cleaning: {len(final_dataset)}")
    print(f"Antal fjernede rækker: {len(dataset) - len(final_dataset)}")
    
    return final_dataset

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
    plt.savefig(f'correlation_heatmap_{CONTINENT}.png')
    print(f"\nKorrelationsmatrix gemt som: correlation_heatmap_{CONTINENT}.png")
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
    numeric_data = dataset[analysis_features].select_dtypes(include=['float64', 'int64'])
    
    # Håndter manglende data
    clean_data = handle_missing_data(numeric_data)
    
    # Indstil Pandas til at vise alle rækker
    pd.set_option('display.max_rows', None)
    
    # Beregn korrelationer
    correlation_matrix = clean_data.corr()
    
    # Print korrelationer med target
    print("\nKorrelationer med CO2-udledning:")
    print(correlation_matrix[TARGET])
    
    # Plot korrelationsmatrix
    plot_correlation_heatmap(correlation_matrix)
    
    return clean_data, correlation_matrix

if __name__ == "__main__":
    analyze_correlations()