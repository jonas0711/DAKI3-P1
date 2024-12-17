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
    data = pd.read_csv(DATA_FILE)
    
    # Indlæs features baseret på FEATURES_SELECTED fra kontrolcenter
    with open('udvalgte_features.json', 'r') as file:
        features_groups = json.load(file)
        clustering_features = features_groups[FEATURES_SELECTED]
    
    # Grupperer efter land og beregner gennemsnit
    country_profiles = data.groupby('country')[clustering_features].mean().reset_index()

    # Dropper rækker med manglende data
    country_profiles = country_profiles.dropna()

    return country_profiles, clustering_features

def scale_features(country_profiles, clustering_features=FEATURES_SELECTED):
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
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=39)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Beregn scores
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(silhouette_avg)
        inertias.append(inertia)
    
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
    plt.savefig('Billeder/cluster_evaluation_metrics.png')
    plt.close()

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

def calculate_cluster_feature_averages(country_profiles, clustering_features, n_clusters):
    """
    Beregner gennemsnitsværdier for hver cluster baseret på features brugt i models.py
    """
    # Indlæs features fra udvalgte_features.json
    with open('udvalgte_features.json', 'r') as file:
        features_groups = json.load(file)
        modeling_features = features_groups[FEATURES_SELECTED]
    
    print(f"Gennemsnitsværdier for features brugt i models.py for {n_clusters} clusters:")
    
    # Udfør clustering for at få cluster labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=39)
    features_scaled = scale_features(country_profiles, clustering_features)
    cluster_labels = kmeans.fit_predict(features_scaled)
    country_profiles[f'Cluster_{n_clusters}'] = cluster_labels
    
    # Beregn og print gennemsnit for hver cluster
    for cluster in range(n_clusters):
        cluster_data = country_profiles[country_profiles[f'Cluster_{n_clusters}'] == cluster]
        
        print(f"\nCluster {cluster} gennemsnit:")
        print("-" * 50)
        for feature in modeling_features:
            if feature in country_profiles.columns:
                avg_value = cluster_data[feature].mean()
                print(f"{feature}: {avg_value:.2f}")
    
    return country_profiles

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
    
    # Tilføj analyse af 6 clusters
    print("Analyse af 6 clusters")
    print_cluster_details(country_profiles, clustering_features, 6)
    
    # Beregn og print feature gennemsnit for 6 clusters
    country_profiles = calculate_cluster_feature_averages(country_profiles, clustering_features, 6)
    
    return country_profiles

if __name__ == "__main__":
    # Kør analysen
    country_profiles = analyze_optimal_clusters()