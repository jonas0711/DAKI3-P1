from kontrolcenter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def analyze_year(data, year, clustering_features):
    """Analyserer data for et specifikt år"""
    # Filtrer data for det specifikke år
    year_data = data[data['year'] == year].copy()
    
    # Fjern eventuelle NaN værdier
    year_data = year_data.dropna(subset=clustering_features)
    
    print(f"\nAnalyse for år {year}")
    print(f"Antal lande med komplette data: {len(year_data)}")
    
    # Standardiser features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(year_data[clustering_features])
    features_scaled = pd.DataFrame(features_scaled, columns=clustering_features)
    
    # Udfør clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=39)
    year_data['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # PCA for visualisering
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features_scaled)
    
    # Plot PCA
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                         c=year_data['Cluster'], cmap='viridis')
    
    # Tilføj landenavne
    for i, country in enumerate(year_data['country']):
        plt.annotate(country, (pca_features[i, 0], pca_features[i, 1]))
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'Landeclusters {year}')
    plt.colorbar(scatter, label='Cluster')
    plt.show()
    
    # Vis cluster-karakteristika med heatmap
    plt.figure(figsize=(12, 8))
    cluster_means = year_data.groupby('Cluster')[clustering_features].mean()
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Karakteristika for hver cluster - {year}')
    plt.show()
    
    # Print cluster information
    for cluster in range(n_clusters):
        print(f"\nCluster {cluster} for {year}:")
        cluster_countries = year_data[year_data['Cluster'] == cluster]['country'].tolist()
        print("Lande:", ", ".join(cluster_countries))
        
        # Beregn gennemsnitlige værdier
        cluster_means = year_data[year_data['Cluster'] == cluster][clustering_features].mean()
        print("\nGennemsnitlige værdier:")
        for feature, value in cluster_means.items():
            print(f"{feature}: {value:.2f}")
    
    # Gem cluster-inddelinger
    cluster_assignments = pd.DataFrame({
        'country': year_data['country'],
        'cluster': year_data['Cluster']
    })
    filename = f'country_clusters_{year}.csv'
    cluster_assignments.to_csv(filename, index=False)
    print(f"\nCluster-inddelinger gemt i '{filename}'")
    
    # Beregn silhouette score
    silhouette_avg = silhouette_score(features_scaled, year_data['Cluster'])
    print(f"Silhouette Score for {year}: {silhouette_avg:.3f}")
    
    return year_data['Cluster'].value_counts().sort_index()

# Indlæs data
data = pd.read_csv(DATA_FILE)

# Definer features
clustering_features = [
    'Value_co2_emissions_kt_by_country',
    'gdp_growth',
    'energy_per_gdp',
    'fossil_share_energy',
    'renewables_share_energy',
    'Primary energy consumption per capita (kWh/person)',
    'gdp_per_capita'
]

# Analyser hvert år
years = [2000, 2010, 2019]
cluster_sizes = {}
for year in years:
    cluster_sizes[year] = analyze_year(data, year, clustering_features)

# Plot udvikling i cluster-størrelser
plt.figure(figsize=(12, 6))
cluster_df = pd.DataFrame(cluster_sizes).fillna(0)
cluster_df.plot(kind='bar')
plt.title('Udvikling i antal lande per cluster')
plt.xlabel('Cluster')
plt.ylabel('Antal lande')
plt.legend(title='År')
plt.show()

# Analyser stabilitet af clusters
# Gem alle clusters i separate DataFrames
clusters_2000 = pd.read_csv('country_clusters_2000.csv').set_index('country')
clusters_2010 = pd.read_csv('country_clusters_2010.csv').set_index('country')
clusters_2019 = pd.read_csv('country_clusters_2019.csv').set_index('country')

# Find lande der er til stede i alle år
common_countries = set(clusters_2000.index) & set(clusters_2010.index) & set(clusters_2019.index)

# Analyser ændringer
changes = pd.DataFrame(index=common_countries)
changes['2000'] = clusters_2000.loc[common_countries]['cluster']
changes['2010'] = clusters_2010.loc[common_countries]['cluster']
changes['2019'] = clusters_2019.loc[common_countries]['cluster']

# Find lande der har skiftet cluster
changed_countries = changes[changes.nunique(axis=1) > 1]
print("\nLande der har skiftet cluster mellem 2000 og 2019:")
for country in changed_countries.index:
    print(f"\n{country}:")
    print(f"2000: Cluster {changes.loc[country, '2000']}")
    print(f"2010: Cluster {changes.loc[country, '2010']}")
    print(f"2019: Cluster {changes.loc[country, '2019']}")

# Gem ændringer til fil
changes.to_csv('cluster_changes_2000_2019.csv')
print("\nÆndringer i clusters gemt i 'cluster_changes_2000_2019.csv'")