from kontrolcenter import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import features
import json

def get_cluster_countries():
    """Henter liste over lande for den valgte cluster"""
    with open(CLUSTERS_FILE, 'r') as f:
        clusters = json.load(f)
    return clusters[SELECTED_CLUSTER]['countries']

def load_and_prepare_data(model_type='gradient_boosting'):
    """
    Indlæser og forbereder test data samt model
    """
    print(f"\nIndlæser data...")
    
    # Hent lande for den valgte cluster
    cluster_countries = get_cluster_countries()
    print(f"\nAnalyserer følgende lande fra {SELECTED_CLUSTER}:")
    print(", ".join(cluster_countries))
    
    # Indlæs og forbered data
    data = features.select_data()
    
    # Filtrer data til kun at inkludere lande fra den valgte cluster
    data = data[data['country'].isin(cluster_countries)]
    
    # Vi vil kun have data fra det seneste år i datasættet
    latest_year = data['year'].max()
    data = data[data['year'] == latest_year]
    print(f"\nBruger data fra år: {latest_year}")
    
    # Split i features og target
    feature_cols = [col for col in data.columns if col not in [TARGET, 'year', 'country']]
    X_test = data[feature_cols]
    y_test = data[TARGET]
    countries_test = data['country']
    
    # Indlæs model baseret på type
    print(f"\nIndlæser {model_type} model...")
    if model_type == 'svr':
        model_filename = f'svr_model_{SELECTED_CLUSTER}.joblib'
        scaler_filename = f'svr_scaler_{SELECTED_CLUSTER}.joblib'
        try:
            model = joblib.load(model_filename)
            scaler = joblib.load(scaler_filename)
            X_test = scaler.transform(X_test)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Kunne ikke finde model eller scaler filer: {str(e)}")
    else:
        model_filename = f'{model_type}_model_{SELECTED_CLUSTER}.joblib'
        try:
            model = joblib.load(model_filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Kunne ikke finde model-filen: {model_filename}")
    
    # Lav forudsigelser
    y_pred = model.predict(X_test)
    
    # Saml resultaterne i en DataFrame
    results = pd.DataFrame({
        'country': countries_test,
        'actual': y_test,
        'predicted': y_pred
    })
    
    return results

def plot_performance(results, model_type='gradient_boosting'):
    """
    Plotter faktiske vs. forudsagte værdier for det seneste år
    """
    plt.figure(figsize=(10, 8))
    
    # Beregn den ideelle linje (y=x)
    min_val = min(results['actual'].min(), results['predicted'].min())
    max_val = max(results['actual'].max(), results['predicted'].max())
    ideal_line = np.linspace(min_val, max_val, 100)
    
    # Plot den ideelle linje
    plt.plot(ideal_line, ideal_line, 'r--', label='Ideel prædiktion', alpha=0.5)
    
    # Plot punkter
    plt.scatter(results['actual'], results['predicted'], alpha=0.6)
    
    # Tilføj labels for hvert land
    for idx, row in results.iterrows():
        plt.annotate(row['country'], 
                    (row['actual'], row['predicted']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10)
    
    # Tilføj labels og titel
    latest_year = results['actual'].name if hasattr(results['actual'], 'name') else 'seneste år'
    plt.xlabel(f'Faktisk CO2-udledning (kt) - {latest_year}')
    plt.ylabel(f'Forudsagt CO2-udledning (kt) - {latest_year}')
    plt.title(f'Sammenligning af Faktiske og Forudsagte CO2-udledninger\n{SELECTED_CLUSTER} - {model_type}')
    
    # Tilføj grid
    plt.grid(True, alpha=0.3)
    
    # Juster layout
    plt.tight_layout()
    
    # Gem plot
    plt.savefig(f'prediction_performance_{model_type}_{SELECTED_CLUSTER}.png', 
                bbox_inches='tight',
                dpi=300)
    print(f"\nPlot gemt som: prediction_performance_{model_type}_{SELECTED_CLUSTER}.png")
    plt.close()

def calculate_metrics(results):
    """
    Beregner performance metrikker for hvert land
    """
    # Beregn fejl for hver observation
    results['error'] = results['predicted'] - results['actual']
    results['error_pct'] = (results['error'] / results['actual']) * 100
    
    # Formatter resultater som en pæn tabel
    metrics = pd.DataFrame({
        'Land': results['country'],
        'Faktisk': results['actual'].round(2),
        'Forudsagt': results['predicted'].round(2),
        'Fejl': results['error'].round(2),
        'Fejl_%': results['error_pct'].round(2)
    })
    
    return metrics

def analyze_predictions(model_type='gradient_boosting'):
    """
    Hovedfunktion der udfører analysen
    """
    # Indlæs og forbered data
    results = load_and_prepare_data(model_type)
    
    # Beregn metrikker
    metrics = calculate_metrics(results)
    
    # Print resultater
    print("\nPerformance metrikker for seneste år:")
    print(metrics.to_string(index=False))
    
    # Lav plot
    plot_performance(results, model_type)
    
    return results, metrics

if __name__ == "__main__":
    # Liste over modeller der skal evalueres
    model_types = ['gradient_boosting', 'random_forest', 'svr']
    
    for model_type in model_types:
        try:
            print(f"\n=== Evaluerer {model_type} model ===")
            analyze_predictions(model_type)
        except Exception as e:
            print(f"Fejl ved evaluering af {model_type}: {str(e)}")