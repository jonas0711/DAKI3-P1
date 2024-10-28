from kontrolcenter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score

def analyze_optimal_clusters(min_clusters=2, max_clusters=20):
   """
   Analyserer det optimale antal clusters ved hjælp af både silhouette score og inertia.
   Tager minimum og maksimum antal clusters som input.
   """
   # Indlæs data fra kontrolcenter
   print(f"\nIndlæser data fra: {DATA_FILE}")
   data = pd.read_csv(DATA_FILE)

   # Vælg features til clustering
   clustering_features = [
       'Value_co2_emissions_kt_by_country',
       'gdp_growth',
       'energy_per_gdp',
       'fossil_share_energy',
       'renewables_share_energy',
       'Primary energy consumption per capita (kWh/person)',
       'gdp_per_capita'
   ]

   # Forbered data
   print("\nForbereder data...")
   country_profiles = data.groupby('country')[clustering_features].mean().reset_index()
   country_profiles = country_profiles.dropna()
   print(f"Antal lande med komplette data: {len(country_profiles)}")

   # Standardiser features
   scaler = StandardScaler()
   features_scaled = scaler.fit_transform(country_profiles[clustering_features])
   
   # Lists til at gemme resultater
   silhouette_scores = []
   inertias = []
   
   # Test forskellige antal clusters
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

   # Find top 3 bedste antal clusters baseret på silhouette score
   top_3_indices = np.argsort(silhouette_scores)[-3:][::-1]
   top_3_n_clusters = [i + min_clusters for i in top_3_indices]
   
   print("\nTop 3 bedste antal clusters baseret på silhouette score:")
   for i, n_clusters in enumerate(top_3_n_clusters):
       print(f"{i+1}. {n_clusters} clusters (Silhouette Score: {silhouette_scores[n_clusters-min_clusters]:.3f})")

   # Plot resultater for alle tre bedste cluster antal
   for n_clusters in top_3_n_clusters:
       kmeans = KMeans(n_clusters=n_clusters, random_state=39)
       cluster_labels = kmeans.fit_predict(features_scaled)
       country_profiles[f'Cluster_{n_clusters}'] = cluster_labels

       # Plot karakteristika for hver cluster
       plt.figure(figsize=(15, 8))
       plt.title(f'Karakteristika for {n_clusters} clusters')
       
       # Beregn gennemsnit for hver cluster
       cluster_means = country_profiles.groupby(f'Cluster_{n_clusters}')[clustering_features].mean()
       
       # Heatmap
       sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt='.2f')
       plt.tight_layout()
       plt.show()

       # Scatter plot af lande
       plt.figure(figsize=(15, 8))
       plt.title(f'Landegruppering med {n_clusters} clusters')
       
       # Vælg to vigtige features til visualisering
       scatter = plt.scatter(
           country_profiles['Primary energy consumption per capita (kWh/person)'],
           country_profiles['renewables_share_energy'],
           c=country_profiles[f'Cluster_{n_clusters}'],
           cmap='viridis'
       )
       
       # Tilføj landenavne
       for i, country in enumerate(country_profiles['country']):
           plt.annotate(country, (
               country_profiles['Primary energy consumption per capita (kWh/person)'].iloc[i],
               country_profiles['renewables_share_energy'].iloc[i]
           ))
       
       plt.xlabel('Energiforbrug per capita (kWh/person)')
       plt.ylabel('Vedvarende energi andel (%)')
       plt.colorbar(scatter, label='Cluster')
       plt.tight_layout()
       plt.show()

       # Print detaljer for hver cluster
       print(f"\nDetaljer for {n_clusters} clusters:")
       for cluster in range(n_clusters):
           cluster_countries = country_profiles[country_profiles[f'Cluster_{n_clusters}'] == cluster]
           print(f"\nCluster {cluster}:")
           print("Lande:", ", ".join(cluster_countries['country'].tolist()))
           print("\nGennemsnitlige værdier:")
           for feature in clustering_features:
               print(f"{feature}: {cluster_countries[feature].mean():.2f}")
           print("-" * 50)

   # Gem resultaterne for alle tre bedste cluster antal
   output_file = 'cluster_results_top_3.csv'
   country_profiles.to_csv(output_file, index=False)
   print(f"\nResultater gemt i: {output_file}")

   return country_profiles, top_3_n_clusters

if __name__ == "__main__":
   # Kør analysen
   country_profiles, top_3_n_clusters = analyze_optimal_clusters()