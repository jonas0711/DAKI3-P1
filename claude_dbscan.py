from kontrolcenter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score

def analyze_dbscan_clusters(eps_range=np.arange(0.1, 2.1, 0.1), min_samples_range=range(2, 11)):
   """
   Analyserer clusters ved hjælp af DBSCAN med forskellige epsilon og min_samples værdier.
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
   results = []
   
   # Test forskellige parametre
   print("\nTester forskellige DBSCAN parametre...")
   for eps in eps_range:
       for min_samples in min_samples_range:
           dbscan = DBSCAN(eps=eps, min_samples=min_samples)
           cluster_labels = dbscan.fit_predict(features_scaled)
           
           # Tæl antal clusters (ekskluder støj points som er -1)
           n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
           n_noise = list(cluster_labels).count(-1)
           
           # Beregn silhouette score hvis der er mere end én cluster og mindre end n-1 støjpunkter
           if n_clusters > 1 and n_noise < len(cluster_labels) - 1:
               # Fjern støjpunkter ved beregning af silhouette score
               mask = cluster_labels != -1
               if len(set(cluster_labels[mask])) > 1:  # Check om der er mere end én cluster efter fjernelse af støj
                   silhouette_avg = silhouette_score(features_scaled[mask], cluster_labels[mask])
               else:
                   silhouette_avg = 0
           else:
               silhouette_avg = 0
           
           results.append({
               'eps': eps,
               'min_samples': min_samples,
               'n_clusters': n_clusters,
               'n_noise': n_noise,
               'silhouette': silhouette_avg
           })
           
           print(f"eps: {eps:.1f}, min_samples: {min_samples}")
           print(f"Antal clusters: {n_clusters}")
           print(f"Antal støjpunkter: {n_noise}")
           print(f"Silhouette Score: {silhouette_avg:.3f}\n")

   # Find de bedste parametre
   results_df = pd.DataFrame(results)
   valid_results = results_df[results_df['n_clusters'] > 1]  # Kun resultater med mere end én cluster
   
   if len(valid_results) > 0:
       best_result = valid_results.loc[valid_results['silhouette'].idxmax()]
       print("\nBedste parametre:")
       print(f"Epsilon: {best_result['eps']}")
       print(f"Min Samples: {best_result['min_samples']}")
       print(f"Antal clusters: {best_result['n_clusters']}")
       print(f"Silhouette Score: {best_result['silhouette']:.3f}")

       # Lav den endelige clustering med de bedste parametre
       best_dbscan = DBSCAN(eps=best_result['eps'], min_samples=int(best_result['min_samples']))
       country_profiles['Cluster'] = best_dbscan.fit_predict(features_scaled)

       # Plot heatmap for clusters
       plt.figure(figsize=(15, 8))
       plt.title(f'Karakteristika for clusters (eps={best_result["eps"]}, min_samples={int(best_result["min_samples"])})')
       
       # Beregn gennemsnit for hver cluster (ekskluder støjpunkter)
       cluster_means = country_profiles[country_profiles['Cluster'] != -1].groupby('Cluster')[clustering_features].mean()
       
       # Heatmap
       sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt='.2f')
       plt.tight_layout()
       plt.show()

       # Scatter plot af lande
       plt.figure(figsize=(15, 8))
       plt.title('Landegruppering med DBSCAN')
       
       # Plot lande
       scatter = plt.scatter(
           country_profiles['Primary energy consumption per capita (kWh/person)'],
           country_profiles['renewables_share_energy'],
           c=country_profiles['Cluster'],
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
       print("\nCluster detaljer:")
       unique_clusters = sorted(set(country_profiles['Cluster']))
       for cluster in unique_clusters:
           cluster_countries = country_profiles[country_profiles['Cluster'] == cluster]
           if cluster == -1:
               print("\nStøjpunkter (outliers):")
           else:
               print(f"\nCluster {cluster}:")
           print("Lande:", ", ".join(cluster_countries['country'].tolist()))
           print("\nGennemsnitlige værdier:")
           for feature in clustering_features:
               print(f"{feature}: {cluster_countries[feature].mean():.2f}")
           print("-" * 50)

       # Gem resultaterne
       output_file = f'dbscan_clusters_eps{best_result["eps"]}_minSamples{int(best_result["min_samples"])}.csv'
       country_profiles.to_csv(output_file, index=False)
       print(f"\nResultater gemt i: {output_file}")

   else:
       print("\nIngen gyldige clusters fundet med de givne parametre.")

   return country_profiles, results_df

if __name__ == "__main__":
   # Kør analysen med standard parametre
   country_profiles, results = analyze_dbscan_clusters()