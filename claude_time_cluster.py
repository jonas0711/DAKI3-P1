from kontrolcenter import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import joblib
import json

def calculate_development_patterns(data, feature_columns):
    """
    Beregner udviklingsmønstre for hver feature for hvert land med fokus på progression over tid
    """
    data = data.sort_values(['country', 'year'])
    patterns_dict = {}

    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        country_patterns = {}

        for feature in feature_columns:
            # Beregn år-til-år ændringer for feature
            yearly_changes = country_data[feature].pct_change(fill_method=None) * 100
            
            # Beregn overordnet trend (lineær regression koefficient)
            years = np.arange(len(country_data))
            values = country_data[feature].values
            if len(years) > 1:
                trend = np.polyfit(years, values, 1)[0]
            else:
                trend = 0
            
            # Beregn acceleration (ændring i ændringsrate)
            acceleration = yearly_changes.diff()
            
            # Håndter NaN og inf værdier
            yearly_changes = yearly_changes.replace([np.inf, -np.inf], np.nan).fillna(0)
            acceleration = acceleration.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Beregn standardafvigelse i udvikling
            stability = 1 - (yearly_changes.std() / (abs(yearly_changes.mean()) + 1e-6))
            
            # Beregn total procentvis ændring
            if country_data[feature].iloc[0] != 0:
                total_change = ((country_data[feature].iloc[-1] / 
                               country_data[feature].iloc[0] - 1) * 100)
            else:
                total_change = 0
            
            # Gem mønstre for denne feature
            country_patterns.update({
                f"{feature}_trend": trend,
                f"{feature}_mean_change": yearly_changes.mean(),
                f"{feature}_change_stability": stability,
                f"{feature}_acceleration": acceleration.mean(),
                f"{feature}_total_change": total_change,
                f"{feature}_volatility": yearly_changes.std()
            })
        
        patterns_dict[country] = country_patterns
    
    patterns_df = pd.DataFrame.from_dict(patterns_dict, orient='index')
    return patterns_df

def prepare_training_data():
    """
    Forbereder data til clustering med fokus på relevante features for CO2-udledning
    """
    print(f"\nIndlæser data fra: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    
    print("\nTilgængelige kolonner:", data.columns.tolist())
    
    # Filtrer data til træningsperiode
    train_data = data[data['year'] <= YEAR_SPLIT]
    
    # Definér feature grupper
    feature_groups = {
        'primary': {
            'features': ['Value_co2_emissions_kt_by_country'],
            'weight': 0.4
        },
        'energy': {
            'features': [
                'Primary energy consumption per capita (kWh/person)',
                'fossil_share_energy',
                'Renewable energy share in the total final energy consumption (%)',
                'energy_per_gdp'
            ],
            'weight': 0.35
        },
        'economic': {
            'features': [
                'gdp_per_capita',
                'gdp_growth'
            ],
            'weight': 0.25
        }
    }
    
    # Fladgør feature liste
    all_features = []
    for group in feature_groups.values():
        all_features.extend(group['features'])
    
    # Beregn udviklingsmønstre
    development_patterns = calculate_development_patterns(train_data, all_features)
    
    # Print information om NaN værdier før imputation
    print("\nManglende værdier før imputation:")
    print(development_patterns.isna().sum())
    print("\nProcent manglende værdier:")
    print((development_patterns.isna().sum() / len(development_patterns) * 100).round(2))
    
    # Håndter manglende værdier før normalisering
    imputer = SimpleImputer(strategy='mean')
    development_patterns_imputed = pd.DataFrame(
        imputer.fit_transform(development_patterns),
        columns=development_patterns.columns,
        index=development_patterns.index
    )
    
    # Normaliser data
    scaler = StandardScaler()
    patterns_scaled = pd.DataFrame(
        scaler.fit_transform(development_patterns_imputed),
        columns=development_patterns_imputed.columns,
        index=development_patterns_imputed.index
    )
    
    # Gem scaler til senere brug
    joblib.dump(scaler, 'scaler.joblib')
    
    print(f"\nAntal lande efter data preparation: {len(patterns_scaled)}")
    
    return patterns_scaled, feature_groups

def evaluate_clustering(data, min_clusters=2, max_clusters=20):
    """
    Evaluerer forskellige antal clusters med både silhouette score og inertia
    """
    print("\nAnalyserer forskellige antal clusters...")
    
    silhouette_scores = []
    inertias = []
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        
        # Beregn scores
        silhouette_avg = silhouette_score(data, labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(silhouette_avg)
        inertias.append(inertia)
        
        print(f"Antal clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Inertia: {inertia:.2f}\n")
    
    return silhouette_scores, inertias

def find_optimal_clusters(development_patterns, min_clusters=2, max_clusters=20):
    """
    Finder optimalt antal clusters baseret på kombinationen af silhouette score og inertia
    """
    silhouette_scores, inertias = evaluate_clustering(development_patterns, min_clusters, max_clusters)
    
    # Konverter lister til numpy arrays
    silhouette_scores = np.array(silhouette_scores)
    inertias = np.array(inertias)
    
    # Normaliser inertia scores
    normalized_inertias = 1 - (inertias / max(inertias))
    
    # Beregn ændring i inertia
    inertia_changes = np.diff(inertias)
    inertia_changes = np.append(inertia_changes, inertia_changes[-1])
    normalized_changes = (inertia_changes - min(inertia_changes)) / (max(inertia_changes) - min(inertia_changes))
    
    # Kombiner scores
    combined_scores = []
    for i in range(len(silhouette_scores)):
        combined_score = (silhouette_scores[i] + normalized_changes[i]) / 2
        combined_scores.append(combined_score)
        
        print(f"\nDetaljeret scoring for {i + min_clusters} clusters:")
        print(f"Silhouette Score: {silhouette_scores[i]:.3f}")
        print(f"Normaliseret Inertia ændring: {normalized_changes[i]:.3f}")
        print(f"Kombineret score: {combined_score:.3f}")
    
    # Visualiser resultater
    plt.figure(figsize=(15, 5))
    
    # Silhouette plot
    plt.subplot(1, 3, 1)
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Antal clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    # Elbow plot
    plt.subplot(1, 3, 2)
    plt.plot(range(min_clusters, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Plot')
    plt.xlabel('Antal clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    # Kombinerede scores plot
    plt.subplot(1, 3, 3)
    plt.plot(range(min_clusters, max_clusters + 1), combined_scores, marker='o')
    plt.title('Kombinerede Scores')
    plt.xlabel('Antal clusters')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_evaluation.png')
    plt.show()

    # Find bedste antal clusters
    best_n_clusters = min_clusters + combined_scores.index(max(combined_scores))
    
    # Træn endelig model
    final_model = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = final_model.fit_predict(development_patterns)
    
    return {
        'n_clusters': best_n_clusters,
        'silhouette_score': silhouette_scores[best_n_clusters-min_clusters],
        'inertia': inertias[best_n_clusters-min_clusters],
        'combined_score': max(combined_scores),
        'labels': labels,
        'model': final_model
    }

def visualize_clusters(data, labels, n_clusters, feature_groups):
    """
    Visualiserer clusters med forbedret indsigt
    """
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = labels
    
    # Plot mønstre for hver feature gruppe
    all_features = data.columns
    for group_name, group_info in feature_groups.items():
        plt.figure(figsize=(15, 8))
        relevant_features = [col for col in all_features 
                           if any(feature in col for feature in group_info['features'])]
        
        for cluster in range(n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            cluster_means = cluster_data[relevant_features].mean()
            plt.plot(range(len(relevant_features)), cluster_means, 
                    label=f'Cluster {cluster}', marker='o')
        
        plt.title(f'Gennemsnitlige mønstre for {group_name} features per cluster')
        plt.xticks(range(len(relevant_features)), relevant_features, rotation=45, ha='right')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'cluster_patterns_{group_name}.png')
        plt.show()

def print_cluster_countries(data, labels, n_clusters):
    """
    Printer detaljeret information om hvert cluster
    """
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = labels
    
    cluster_stats = {}
    for cluster in range(n_clusters):
        print(f"\nCluster {cluster}:")
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        countries = cluster_data.index.tolist()
        
        print(f"Antal lande: {len(countries)}")
        print("Lande:", ", ".join(countries))
        
        # Beregn og vis nøglestatistikker
        stats = cluster_data.mean()
        print("\nGennemsnitlige karakteristika:")
        for feature in data.columns:
            print(f"{feature}: {stats[feature]:.3f}")
        
        cluster_stats[cluster] = {
            'size': len(countries),
            'countries': countries,
            'stats': stats.to_dict()
        }
        print("-" * 50)
    
    with open('cluster_statistics.json', 'w') as f:
        json.dump(cluster_stats, f, indent=4)

def analyze_development_clusters():
    """
    Hovedfunktion der udfører den komplette analyse
    """
    development_patterns, feature_groups = prepare_training_data()
    
    best_result = find_optimal_clusters(development_patterns)
    
    print(f"\nBedste clustering resultat:")
    print(f"Antal clusters: {best_result['n_clusters']}")
    print(f"Silhouette Score: {best_result['silhouette_score']:.3f}")
    print(f"Inertia: {best_result['inertia']:.2f}")
    print(f"Kombineret score: {best_result['combined_score']:.3f}")
    
    visualize_clusters(development_patterns, best_result['labels'], 
                      best_result['n_clusters'], feature_groups)
    print_cluster_countries(development_patterns, best_result['labels'], 
                          best_result['n_clusters'])
    
    joblib.dump(best_result['model'], 'best_clustering_model.joblib')
    
    development_patterns['Cluster'] = best_result['labels']
    development_patterns.to_csv(f'development_clusters_{YEAR_SPLIT}.csv')
    
    print(f"\nAlle resultater og modeller er gemt")
    return development_patterns, best_result

if __name__ == "__main__":
    development_patterns, best_clustering = analyze_development_clusters()