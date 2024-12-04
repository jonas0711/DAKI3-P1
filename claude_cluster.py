from kontrolcenter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
import json

def load_and_prepare_data():
    """
    Indlæser og forbereder data til clustering analyse
    """
    print(f"\nIndlæser data fra: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    
    # Indlæs features baseret på FEATURES_SELECTED fra kontrolcenter
    with open('udvalgte_features.json', 'r') as file:
        features_groups = json.load(file)
        clustering_features = features_groups[FEATURES_SELECTED]
    
    # Fjern 'country' og 'year' hvis de findes i listen
    clustering_features = [f for f in clustering_features if f not in ['country', 'year']]
    
    print(f"\nBruger features fra {FEATURES_SELECTED}:")
    print(clustering_features)
    
    # Verificer at alle features findes i datasættet
    available_features = [f for f in clustering_features if f in data.columns]
    missing_features = [f for f in clustering_features if f not in data.columns]
    
    if missing_features:
        print("\nAdvarsel: Følgende features blev ikke fundet i datasættet:")
        print(missing_features)
        print("\nBruger kun tilgængelige features:")
        print(available_features)
    
    # Grupperer efter land og beregner gennemsnit
    country_profiles = data.groupby('country')[available_features].mean().reset_index()

    # Dropper rækker med manglende data
    country_profiles = country_profiles.dropna()
    print(f"\nAntal lande med komplette data: {len(country_profiles)}")

    return country_profiles, available_features

def scale_features(country_profiles, clustering_features):
    """
    Standardiserer features til clustering
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(country_profiles[clustering_features])
    return features_scaled

def evaluate_clusters(features_scaled, min_clusters, max_clusters):
    """
    Evaluerer forskellige antal clusters og returnerer scores
    """
    silhouette_scores = []
    inertias = []
    
    print("\nTester forskellige antal clusters...")
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=39)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Beregn scores
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(silhouette_avg)
        inertias.append(inertia)
        
        print(f"Antal clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Inertia: {inertia:.2f}\n")
    
    return silhouette_scores, inertias

def plot_evaluation_metrics(silhouette_scores, inertias, min_clusters, max_clusters):
    """
    Plotter inertia og silhouette scores side om side
    """
    # Opret figure med to subplots side om side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # X-akse værdier (antal clusters)
    clusters_range = list(range(min_clusters, max_clusters + 1))
    
    # Plot inertia (elbow curve)
    ax1.plot(clusters_range, inertias, marker='o', linestyle='-', color='blue')
    ax1.set_title('Elbow Curve (Inertia)')
    ax1.set_xlabel('Antal Clusters')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    # Sæt x-aksen til kun at vise hele tal
    ax1.set_xticks(clusters_range)
    
    # Plot silhouette scores
    ax2.plot(clusters_range, silhouette_scores, marker='o', linestyle='-', color='red')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('Antal Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)
    # Sæt x-aksen til kun at vise hele tal
    ax2.set_xticks(clusters_range)
    
    # Juster layout og gem plot
    plt.tight_layout()
    plt.savefig('cluster_evaluation_metrics.png')
    plt.close()

def find_best_clusters(silhouette_scores, min_clusters):
    """
    Finder de tre bedste antal clusters baseret på silhouette score
    """
    # Sortere silhouette_scores fra højest til lavest
    top_3_indices = np.argsort(silhouette_scores)[-3:][::-1]
    top_3_n_clusters = [i + min_clusters for i in top_3_indices]
    
    print("\nTop 3 bedste antal clusters baseret på silhouette score:")
    for i, n_clusters in enumerate(top_3_n_clusters):
        print(f"{i+1}. {n_clusters} clusters (Silhouette Score: {silhouette_scores[n_clusters-min_clusters]:.3f})")
    
    return top_3_n_clusters

def print_cluster_details(country_profiles, clustering_features, n_clusters):
    """
    Printer detaljeret information om hvert cluster
    """
    # Udfør clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=39)
    features_scaled = scale_features(country_profiles, clustering_features)
    cluster_labels = kmeans.fit_predict(features_scaled)
    country_profiles[f'Cluster_{n_clusters}'] = cluster_labels
    
    print(f"\nDetaljer for {n_clusters} clusters:")
    for cluster in range(n_clusters):
        cluster_countries = country_profiles[country_profiles[f'Cluster_{n_clusters}'] == cluster]
        
        print(f"\nCluster {cluster}:")
        print("Lande:", ", ".join(cluster_countries['country'].tolist()))
        print("\nGennemsnitlige værdier:")
        for feature in clustering_features:
            print(f"{feature}: {cluster_countries[feature].mean():.2f}")
        print("-" * 50)

def analyze_optimal_clusters(min_clusters=2, max_clusters=20):
    """
    Hovedfunktion der udfører den complette cluster analyse
    """
    # Indlæs og forbered data
    country_profiles, clustering_features = load_and_prepare_data()
    
    # Standardiser features
    features_scaled = scale_features(country_profiles, clustering_features)
    
    # Evaluer forskellige antal clusters
    silhouette_scores, inertias = evaluate_clusters(features_scaled, min_clusters, max_clusters)
    
    # Plot evalueringsmetrikker
    plot_evaluation_metrics(silhouette_scores, inertias, min_clusters, max_clusters)
    
    # Find de bedste antal clusters
    top_3_n_clusters = find_best_clusters(silhouette_scores, min_clusters)
    
    # Print detaljer for de bedste cluster løsninger
    for n_clusters in top_3_n_clusters:
        print_cluster_details(country_profiles, clustering_features, n_clusters)
    
    # Gem resultater
    output_file = 'cluster_results_top_3.csv'
    country_profiles.to_csv(output_file, index=False)
    print(f"\nResultater gemt i: {output_file}")

    return country_profiles, top_3_n_clusters

if __name__ == "__main__":
    # Kør analysen
    country_profiles, top_3_n_clusters = analyze_optimal_clusters()