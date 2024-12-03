from kontrolcenter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
import features
import json

def load_and_prepare_data():
    """
    Indlæser og forbereder data til clustering analyse
    """
    print(f"\nIndlæser data fra: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)

    train_data, test_data = features.split_data(data)

    with open('udvalgte_features.json', 'r') as file:
        features_groups = json.load(file)
        features = features_groups[FEATURES_SELECTED]

    # Grupperer land og laver gns på "Clustering_features" og sætter det som index
    country_profiles = train_data.groupby('country')[features].mean().reset_index()

    # Dropper den kolonne med manglende data
    country_profiles = country_profiles.dropna()
    print(f"Antal lande med komplette data: {len(country_profiles)}")

    return country_profiles, features

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
        
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(silhouette_avg)
        inertias.append(inertia)
        
        print(f"Antal clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Inertia: {inertia:.2f}\n")
    
    return silhouette_scores, inertias

def find_best_clusters(silhouette_scores, min_clusters):
    """
    Finder de tre bedste antal clusters baseret på silhouette score
    """
    # Sortere silhouette_scores fra lavest til højest og slicer til de 3 højeste
    top_3_indices = np.argsort(silhouette_scores)[-3:][::-1]

    # Finder det faktiske index nr.
    top_3_n_clusters = [i + min_clusters for i in top_3_indices]
    
    print("\nTop 3 bedste antal clusters baseret på silhouette score:")
    for i, n_clusters in enumerate(top_3_n_clusters):
        print(f"{i+1}. {n_clusters} clusters (Silhouette Score: {silhouette_scores[n_clusters-min_clusters]:.3f})")
    
    return top_3_n_clusters

def plot_cluster_characteristics(country_profiles, clustering_features, n_clusters):
    """
    Plotter karakteristika for et givet antal clusters
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=39)
    features_scaled = scale_features(country_profiles, clustering_features)
    cluster_labels = kmeans.fit_predict(features_scaled)
    country_profiles[f'Cluster_{n_clusters}'] = cluster_labels

    # Plot heatmap hvor vi ser de gennemsnitlige værdier for hver cluster for at kunne sammenligne indellinger for clustrene
    plt.figure(figsize=(15, 8))
    plt.title(f'Karakteristika for {n_clusters} clusters')
    cluster_means = country_profiles.groupby(f'Cluster_{n_clusters}')[clustering_features].mean()
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt='.2f')
    plt.tight_layout()
    plt.show()

    # Plot scatter
    plot_scatter_clusters(country_profiles, n_clusters)

def plot_scatter_clusters(country_profiles, n_clusters):
    """
    Laver scatter plot af landene i deres respektive clusters
    """
    plt.figure(figsize=(15, 8))
    plt.title(f'Landegruppering med {n_clusters} clusters')
    
    scatter = plt.scatter(
        country_profiles['Primary energy consumption per capita (kWh/person)'],
        country_profiles['renewables_share_energy'],
        c=country_profiles[f'Cluster_{n_clusters}'],
        cmap='viridis'
    )
    
    for i, country in enumerate(country_profiles['country']):
        # iloc finder værdi i dataframe baseret på position/index
        plt.annotate(country, (
            country_profiles['Primary energy consumption per capita (kWh/person)'].iloc[i],
            country_profiles['renewables_share_energy'].iloc[i]
        ))
    
    plt.xlabel('Energiforbrug per capita (kWh/person)')
    plt.ylabel('Vedvarende energi andel (%)')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()

def print_cluster_details(country_profiles, clustering_features, n_clusters):
    """
    Printer detaljeret information om hver cluster herunder lande, gennemsnitlige featur værdier for klusteret
    Altså det der kendetegner de fundne klustre
    """
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
    Hovedfunktion der udfører den komplette cluster analyse
    """
    # Indlæs og forbered data
    country_profiles, clustering_features = load_and_prepare_data()
    
    # Standardiser features
    features_scaled = scale_features(country_profiles, clustering_features)
    
    # Evaluer forskellige antal clusters
    silhouette_scores, inertias = evaluate_clusters(features_scaled, min_clusters, max_clusters)
    
    # Find de bedste antal clusters
    top_3_n_clusters = find_best_clusters(silhouette_scores, min_clusters)
    
    # Visualiser og analyser de bedste cluster løsninger
    for n_clusters in top_3_n_clusters:
        plot_cluster_characteristics(country_profiles, clustering_features, n_clusters)
        print_cluster_details(country_profiles, clustering_features, n_clusters)
    
    # Gem resultater
    output_file = 'cluster_results_top_3.csv'
    country_profiles.to_csv(output_file, index=False)
    print(f"\nResultater gemt i: {output_file}")

    return country_profiles, top_3_n_clusters

if __name__ == "__main__":
    # Kør analysen
    country_profiles, top_3_n_clusters = analyze_optimal_clusters()