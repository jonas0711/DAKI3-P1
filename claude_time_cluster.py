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
from sklearn.linear_model import LinearRegression

def calculate_development_patterns(data, feature_columns):
    """
    Beregner udviklingsmønstre for hver feature for hvert land med fokus på progression over tid
    """
    data = data.sort_values(['country', 'year'])
    patterns_dict = {}

    for country in data['country'].unique(): # Gennemgår alle lande i datasættet
        # Får alt data omkring bestemt land i 2000-2009 (2010-2020 sorteret fra i prepare_data())
        country_data = data[data['country'] == country]
        country_patterns = {}

        for feature in feature_columns: # Gennemgår alle udvalgte features
            # Beregn år-til-år ændringer for feature
            yearly_changes = country_data[feature].pct_change(fill_method=None) * 100
            
            # Beregn overordnet trend (lineær regression koefficient)
            years = np.arange(len(country_data)) # Længden af listen af lande, lavet om til tal 0-9
            values = country_data[feature].values # Får værdierne for specifik feature
            if len(years) > 1:  # Sikrer at vi har nok data til regression
                trend = np.polyfit(years, values, 1)[0] # Lineær linje for at finde en hældning = trend
            else:
                trend = 0
            
            # Beregn acceleration (ændring i ændringsrate)
            acceleration = yearly_changes.diff()
            
            # Håndter NaN og inf (kommer af division med 0) værdier erstatter med 0
            yearly_changes = yearly_changes.replace([np.inf, -np.inf], np.nan).fillna(0)
            acceleration = acceleration.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Beregn standardafvigelse i udvikling (hvor meget den afviger fra gennemsnitsændringen) høj = stor variation, lav = stabilitet
            stability = 1 - (yearly_changes.std() / (abs(yearly_changes.mean()) + 1e-6)) # abs = absolutte værdi (så ikke negativ), 1e-6 så ikke dividere med 0
            
            # Beregn total procentvis ændring mellem featureværdi i 2000 og 2009
            if country_data[feature].iloc[0] != 0:
                total_change = ((country_data[feature].iloc[-1] / 
                               country_data[feature].iloc[0] - 1) * 100)
            else:
                total_change = 0
            
            # Gem alle mønstre for denne feature
            country_patterns.update({
                f"{feature}_trend": trend,
                f"{feature}_mean_change": yearly_changes.mean(),
                f"{feature}_change_stability": stability,
                f"{feature}_acceleration": acceleration.mean(),
                f"{feature}_total_change": total_change,
                f"{feature}_volatility": yearly_changes.std()
            })
        
        # dict over dictornaries for hvert land, hvori der er trend, change, stability, acc osv. for alle features
        patterns_dict[country] = country_patterns
    
    patterns_df = pd.DataFrame.from_dict(patterns_dict, orient='index')
    return patterns_df

def prepare_training_data():
    """
    Forbereder data til clustering med fokus på relevante features for CO2-udledning
    """
    print(f"\nIndlæser data fra: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    
    # Print kolonner for verifikation
    print("\nTilgængelige kolonner:", data.columns.tolist())
    
    # Filtrer data til træningsperiode
    train_data = data[data['year'] <= YEAR_SPLIT]
    
    # Definér feature grupper med vægte
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
    
    # Valider features
    missing_features = [f for f in all_features if f not in data.columns]
    if missing_features:
        print(f"Advarsel: Følgende features mangler og vil blive udeladt: {missing_features}")
        # Fjern manglende features fra grupperne
        for group in feature_groups.values():
            group['features'] = [f for f in group['features'] if f not in missing_features]
        all_features = [f for f in all_features if f not in missing_features]
    
    if not all_features:
        raise ValueError("Ingen gyldige features tilbage efter validering")
    
    # Beregn udviklingsmønstre
    development_patterns = calculate_development_patterns(train_data, all_features)
    
    # Håndter manglende værdier
    imputer = SimpleImputer(strategy='mean')
    patterns_imputed = pd.DataFrame(
        imputer.fit_transform(development_patterns),
        columns=development_patterns.columns,
        index=development_patterns.index
    )
    
    # Normaliser data med vægte
    scaler = StandardScaler()
    patterns_scaled = pd.DataFrame(
        scaler.fit_transform(patterns_imputed),
        columns=patterns_imputed.columns,
        index=patterns_imputed.index
    )
    
    # Gem scalere til senere brug
    joblib.dump(imputer, 'imputer.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return patterns_scaled, feature_groups

def evaluate_clustering(data, labels, n_clusters):
    """
    Evaluerer clustering kvalitet med multiple metrics
    """
    if len(np.unique(labels)) <= 1:
        return -np.inf
    
    # Beregn basis silhouette score
    silhouette = silhouette_score(data, labels)
    
    # Beregn cluster størrelse balance
    cluster_sizes = np.array([sum(labels == i) for i in range(n_clusters)])
    size_variation = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Beregn cluster kompakthed
    cluster_densities = []
    for i in range(n_clusters):
        cluster_data = data[labels == i]
        if len(cluster_data) > 1:
            # Beregn gennemsnitlig afstand til cluster centroid
            centroid = cluster_data.mean(axis=0)
            distances = np.linalg.norm(cluster_data - centroid, axis=1)
            cluster_densities.append(distances.mean())
    
    density_score = 1 - (np.std(cluster_densities) / np.mean(cluster_densities)) if cluster_densities else 0
    
    # Kombiner scores med vægte
    combined_score = (
        silhouette * 0.5 +
        (1 - size_variation) * 0.3 +
        density_score * 0.2
    )
    
    return combined_score

def find_optimal_clusters(data, min_clusters=2, max_clusters=20):
    """
    Finder optimalt antal clusters med omfattende evaluering
    """
    print("\nAnalyserer forskellige antal clusters...")
    
    results = []
    evaluation_scores = []
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        best_score = -np.inf
        best_labels = None
        best_model = None
        
        # Kør multiple initialiseringer for hver n_clusters
        for _ in range(5):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42+_, n_init=10)
            labels = kmeans.fit_predict(data)
            score = evaluate_clustering(data, labels, n_clusters)
            
            if score > best_score:
                best_score = score
                best_labels = labels
                best_model = kmeans
        
        evaluation_scores.append(best_score)
        results.append({
            'n_clusters': n_clusters,
            'score': best_score,
            'labels': best_labels,
            'model': best_model
        })
        
        print(f"Antal clusters: {n_clusters}, Score: {best_score:.3f}")
    
    # Visualiser scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(min_clusters, max_clusters + 1), evaluation_scores, marker='o')
    plt.title('Cluster Evaluerings Score')
    plt.xlabel('Antal clusters')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig('cluster_scores.png')
    plt.show()
    
    return results

def visualize_clusters(data, labels, n_clusters, feature_groups):
    """
    Visualiserer clusters med forbedret indsigt
    """
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = labels
    
    # Plot gennemsnitlige mønstre per cluster
    plt.figure(figsize=(15, 10))
    
    # Organiser features efter grupper
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
    Printer detaljeret cluster analyse
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
        
        # Gem statistik
        cluster_stats[cluster] = {
            'size': len(countries),
            'countries': countries,
            'stats': stats.to_dict()
        }
        print("-" * 50)
    
    # Gem cluster statistik
    with open('cluster_statistics.json', 'w') as f:
        json.dump(cluster_stats, f, indent=4)

def analyze_development_clusters():
    """
    Hovedfunktion der udfører den komplette analyse
    """
    # Forbered data
    development_patterns, feature_groups = prepare_training_data()
    
    # Find optimale clusters
    cluster_results = find_optimal_clusters(development_patterns)
    
    # Vælg bedste resultat
    best_result = max(cluster_results, key=lambda x: x['score'])
    
    print(f"\nBedste clustering resultat:")
    print(f"Antal clusters: {best_result['n_clusters']}")
    print(f"Score: {best_result['score']:.3f}")
    
    # Visualiser og analyser resultater
    visualize_clusters(development_patterns, best_result['labels'], 
                      best_result['n_clusters'], feature_groups)
    print_cluster_countries(development_patterns, best_result['labels'], 
                          best_result['n_clusters'])
    
    # Gem clustering model og resultater
    joblib.dump(best_result['model'], 'best_clustering_model.joblib')
    
    # Gem udviklingsmønstre med cluster labels
    output_file = f'development_clusters_{YEAR_SPLIT}.csv'
    development_patterns['Cluster'] = best_result['labels']
    development_patterns.to_csv(output_file)
    
    print(f"\nAlle resultater og modeller er gemt")
    return development_patterns, best_result

if __name__ == "__main__":
    development_patterns, best_clustering = analyze_development_clusters()