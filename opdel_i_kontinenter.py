import pandas as pd
import numpy as np

# Opret en dictionary for clusters baseret på vores analyse
country_to_cluster = {
    # Udviklingslande
    'Belarus': 'developing', 'Bulgaria': 'developing', 
    'Croatia': 'developing', 'Estonia': 'developing', 
    'Hungary': 'developing', 'Latvia': 'developing', 
    'Lithuania': 'developing', 'North Macedonia': 'developing', 
    'Poland': 'developing', 'Romania': 'developing', 
    'Ukraine': 'developing', 'Azerbaijan': 'developing', 
    'India': 'developing', 'Indonesia': 'developing', 
    'Iraq': 'developing', 'Israel': 'developing',
    'Kazakhstan': 'developing', 'Malaysia': 'developing', 
    'Pakistan': 'developing', 'Philippines': 'developing',
    'Thailand': 'developing', 'Uzbekistan': 'developing',
    'Argentina': 'developing', 'Chile': 'developing', 
    'Colombia': 'developing', 'Ecuador': 'developing', 
    'Mexico': 'developing', 'Peru': 'developing',
    'Algeria': 'developing', 'Morocco': 'developing', 
    'South Africa': 'developing',
    
    # Ressourcerige
    'Norway': 'resource_rich', 'Sweden': 'resource_rich', 
    'Iceland': 'resource_rich', 'Canada': 'resource_rich',
    'Trinidad and Tobago': 'resource_rich',
    
    # Udviklede
    'Austria': 'developed', 'Belgium': 'developed', 
    'Denmark': 'developed', 'France': 'developed', 
    'Germany': 'developed', 'Greece': 'developed', 
    'Ireland': 'developed', 'Italy': 'developed', 
    'Luxembourg': 'developed', 'Netherlands': 'developed',
    'Portugal': 'developed', 'Slovenia': 'developed', 
    'Spain': 'developed', 'Switzerland': 'developed',
    'United Kingdom': 'developed', 'Australia': 'developed',
    'Japan': 'developed', 'Finland': 'developed', 
    'New Zealand': 'developed',
    
    # Storskala
    'China': 'large_scale', 'United States': 'large_scale'
}

def clean_country_name(country):
    """Renser landenavne for eventuelle ekstra data"""
    if ',' in str(country):
        return country.split(',')[0]
    return country

def process_data():
    """
    Indlæser, processerer og gemmer data i separate filer for hver cluster
    """
    # Læs CSV-filen
    print("Indlæser datasæt...")
    df = pd.read_csv(r'CSV_files/Energi_Data.csv')

    # Rens landenavne
    df['country'] = df['country'].apply(clean_country_name)

    # Tilføj en kolonne til DataFrame, der repræsenterer clusters
    df['cluster'] = df['country'].map(country_to_cluster)

    # Gem en kopi af hele datasættet som world_data.csv
    df.to_csv('CSV_files/world_data.csv', index=False)
    print(f"Data for hele verden gemt i filen: CSV_files/world_data.csv")

    # Find lande der mangler cluster-tildeling
    missing_countries = df[df['cluster'].isna()]['country'].unique()
    if len(missing_countries) > 0:
        print("\nADVARSEL: Følgende lande mangler cluster-tildeling:")
        for country in sorted(missing_countries):
            if pd.notna(country):  # Tjek om landet ikke er NaN
                print(f"'{country}'")

    # Definér cluster rækkefølge
    cluster_order = ['developing', 'resource_rich', 'developed', 'large_scale']

    # Loop igennem hver cluster i den definerede rækkefølge
    for cluster in cluster_order:
        # Filtrer data for den pågældende cluster
        df_cluster = df[df['cluster'] == cluster]
        
        if len(df_cluster) > 0:  # Kun hvis der er data for denne cluster
            # Opret et filnavn baseret på clusterets navn
            file_name = f"CSV_files/cluster_{cluster}_data.csv"
            
            # Gem data som CSV-fil
            df_cluster.to_csv(file_name, index=False)
            
            # Print information om cluster
            print(f"\nCluster: {cluster}")
            unique_countries = sorted(df_cluster['country'].unique())
            print(f"Antal lande: {len(unique_countries)}")
            print(f"Antal datapunkter: {len(df_cluster)}")
            print(f"Data gemt i: {file_name}")
            print("Inkluderede lande:")
            print(", ".join(unique_countries))

def print_usage_guide():
    """
    Printer en guide til hvordan man bruger de genererede filer med kontrolcenter.py
    """
    print("\nBrug af filer med kontrolcenter.py:")
    print("I kontrolcenter.py kan du nu bruge følgende indstillinger:")
    print("\nFor alle lande:")
    print('CONTINENT = "world"')
    print('DATA_FILE = "CSV_files/world_data.csv"')
    print("\nFor udviklingslande:")
    print('CONTINENT = "developing"')
    print('DATA_FILE = "CSV_files/cluster_developing_data.csv"')
    print("\nFor ressourcerige lande:")
    print('CONTINENT = "resource_rich"')
    print('DATA_FILE = "CSV_files/cluster_resource_rich_data.csv"')
    print("\nFor udviklede lande:")
    print('CONTINENT = "developed"')
    print('DATA_FILE = "CSV_files/cluster_developed_data.csv"')
    print("\nFor storskala lande:")
    print('CONTINENT = "large_scale"')
    print('DATA_FILE = "CSV_files/cluster_large_scale_data.csv"')

if __name__ == "__main__":
    process_data()
    print_usage_guide()