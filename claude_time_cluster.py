from kontrolcenter import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from models import randomforestregression, gradientboost
import joblib
import json

def calculate_yearly_changes(data, feature_columns):
    """
    Beregner årlige ændringer for hver feature for hvert land, inkl. acceleration og tidsjusteret standardafvigelse
    """
    # Sorter data efter land og år
    data = data.sort_values(['country', 'year'])
    
    changes_dict = {}
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        
        # Beregn årlige ændringer for hver feature
        yearly_changes = {}
        for feature in feature_columns:
            # Beregn procentvis ændring
            changes = country_data[feature].pct_change(fill_method=None)
            acceleration = changes.diff()  # Beregn acceleration/deceleration
            
            # Håndter NaN og uendelige værdier
            changes = changes.replace([np.inf, -np.inf], np.nan).fillna(0)
            acceleration = acceleration.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Gem statistik om ændringer og accelerationer
            yearly_changes[f"{feature}_mean_change"] = changes.mean()
            yearly_changes[f"{feature}_std_change"] = changes.std()
            yearly_changes[f"{feature}_acceleration_mean"] = acceleration.mean()
            yearly_changes[f"{feature}_acceleration_std"] = acceleration.std()
            yearly_changes[f"{feature}_start"] = country_data[feature].iloc[0]
            yearly_changes[f"{feature}_end"] = country_data[feature].iloc[-1]
            
        changes_dict[country] = yearly_changes
    
    return pd.DataFrame.from_dict(changes_dict, orient='index')

def prepare_training_data():
    """
    Forbereder data til clustering baseret på udviklingsmønstre
    """
    print(f"\nIndlæser data fra: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    
    # Filtrer data til træningsperiode
    train_data = data[data['year'] <= YEAR_SPLIT]
    
    # Vælg relevante features for clustering
    features = [
        'Value_co2_emissions_kt_by_country',
        'gdp_growth',
        'energy_per_gdp',
        'fossil_share_energy',
        'renewables_share_energy',
        'Primary energy consumption per capita (kWh/person)',
        'gdp_per_capita'
    ]
    
    # Beregn udviklingsmønstre med de nye beregninger
    development_patterns = calculate_yearly_changes(train_data, features)
    
    # Imputer manglende værdier med gennemsnit
    imputer = SimpleImputer(strategy='mean')
    development_patterns_imputed = imputer.fit_transform(development_patterns)
    
    # Normaliser dataen
    scaler = StandardScaler()
    development_patterns_scaled = scaler.fit_transform(development_patterns_imputed)
    
    # Konverter tilbage til DataFrame for bedre håndtering senere
    development_patterns_df = pd.DataFrame(development_patterns_scaled, columns=development_patterns.columns, index=development_patterns.index)
    
    return development_patterns_df, features

def find_optimal_clusters(data, min_clusters=2, max_clusters=20):
    """
    Finder det optimale antal clusters baseret på silhouette score
    """
    print("\nTester forskellige antal clusters...")
    
    results = []
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=39)
        cluster_labels = kmeans.fit_predict(data)
        
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette_avg,
            'labels': cluster_labels
        })
        
        print(f"Antal clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")
    
    # Visualiser Silhouette-scorer for at finde det bedste antal clusters
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score per antal clusters')
    plt.xlabel('Antal clusters')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.show()
    
    return results

def visualize_clusters(data, labels, n_clusters):
    """
    Visualiserer clusters og deres udviklingsmønstre
    """
    # Tilføj cluster labels til data
    data['Cluster'] = labels
    
    # Plot gennemsnitlige ændringer for hver cluster
    plt.figure(figsize=(15, 8))
    for cluster in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster].drop(columns=['Cluster'])
        plt.plot(cluster_data.mean(), label=f'Cluster {cluster}')
    
    plt.title('Gennemsnitlige udviklingsmønstre per cluster')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_cluster_countries(data, labels, n_clusters):
    """
    Printer lande i hver cluster og deres karakteristika
    """
    data['Cluster'] = labels
    
    for cluster in range(n_clusters):
        print(f"\nCluster {cluster}:")
        cluster_countries = data[data['Cluster'] == cluster].index
        print("Lande:", ", ".join(cluster_countries))
        
        print("\nGennemsnitlige ændringer:")
        print(data[data['Cluster'] == cluster].mean())
        print("-" * 50)

def analyze_development_clusters():
    """
    Hovedfunktion der udfører hele analysen
    """
    # Forbered data
    development_patterns, feature_list = prepare_training_data()
    
    # Find optimale clusters
    cluster_results = find_optimal_clusters(development_patterns)
    
    # Vælg bedste antal clusters baseret på højeste silhouette score
    best_result = max(cluster_results, key=lambda x: x['silhouette'])
    
    print(f"\nBedste antal clusters: {best_result['n_clusters']}")
    print(f"Silhouette Score: {best_result['silhouette']:.3f}")
    
    # Visualiser og analyser bedste clustering
    visualize_clusters(development_patterns, best_result['labels'], best_result['n_clusters'])
    print_cluster_countries(development_patterns, best_result['labels'], best_result['n_clusters'])
    
    # Gem cluster resultater
    output_file = f'development_clusters_{YEAR_SPLIT}.csv'
    development_patterns['Cluster'] = best_result['labels']
    development_patterns.to_csv(output_file, index=True)
    print(f"\nResultater gemt i: {output_file}")
    
    return development_patterns, best_result

if __name__ == "__main__":
    development_patterns, best_clustering = analyze_development_clusters()
