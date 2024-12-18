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
    return clusters[SELECTED_CLUSTER]

def load_and_prepare_data():
    """
    Indlæser og forbereder test data samt model
    """
    
    # Hent lande for den valgte cluster
    cluster_countries = get_cluster_countries()

    # Indlæs og forbered data
    data = features.select_data()
    
    # Filtrer data til kun at inkludere lande fra den valgte cluster
    data = data[data['country'].isin(cluster_countries)]
    
    # Vi vil kun have data fra 2019
    data = data[data['year'] == 2019]

    # Hent features fra kontrolcenter
    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    feature_cols = feature_schema[FEATURES_SELECTED]
   
    # Split i features og target
    X_test = data[feature_cols]
    y_test = data[TARGET]
    countries_test = data['country']
    
    # Indlæs gradient boosting model
    model_filename = f'Modeller/gradient_boosting_model_{SELECTED_CLUSTER}.joblib'
    try:
        model = joblib.load(model_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Kunne ikke finde model-filen")
    
    # Lav forudsigelser
    y_pred = model.predict(X_test)
    
    # Saml resultaterne i en DataFrame
    results = pd.DataFrame({
        'country': countries_test,
        'actual': y_test,
        'predicted': y_pred
    })
    
    return results

def plot_performance(results):
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
    plt.title(f'Sammenligning af Faktiske og Forudsagte CO2-udledninger\n{SELECTED_CLUSTER}')
    
    # Tilføj grid
    plt.grid(True, alpha=0.3)
    
    # Juster layout
    plt.tight_layout()
    
    # Gem plot
    plt.savefig(f'Performance_png/prediction_performance_{SELECTED_CLUSTER}.png', 
                bbox_inches='tight',
                dpi=300)
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

def analyze_predictions():
    """
    Hovedfunktion der udfører analysen
    """
    # Indlæs og forbered data
    results = load_and_prepare_data()
    
    # Beregn metrikker
    metrics = calculate_metrics(results)
    
    # Print resultater
    print("\nPerformance metrikker for 2019:")
    print(metrics.to_string(index=False))
    
    # Lav plot
    plot_performance(results)
    
    return results, metrics

if __name__ == "__main__":
    try:
        print("Evaluerer gradient boosting model")
        analyze_predictions()
    except Exception as e:
        print(f"Fejl ved evaluering: {str(e)}")