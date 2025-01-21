import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Indlæs data fra filerne
performance_file = 'performance.json'
world_data_file = 'CSV_Files/world_data.csv'

def load_performance_data(filepath, cluster=None):
    """Indlæser og behandler performance data fra en JSON-fil, filtreret efter cluster."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if cluster is not None:
        cluster_key = f"cluster_{cluster}"
        if cluster_key not in data:
            print(f"Fejl: Kunne ikke finde cluster {cluster} i JSON-filen.")
            return pd.DataFrame()

        cluster_data = data[cluster_key]
        simplified_data = [
            {"Land": land.strip().lower(), "GB Fejl%": float(error.replace('%', '').strip())}
            for land, error in cluster_data.items()
        ]
        return pd.DataFrame(simplified_data)
    else:
        print("Fejl: Der skal angives et cluster for at indlæse data.")
        return pd.DataFrame()

def load_world_data(filepath):
    """Indlæser world data fra en CSV-fil og filtrerer for året 2019."""
    data = pd.read_csv(filepath)
    data = data[data['year'] == 2019]  # Filtrer kun for året 2019
    data['country'] = data['country'].str.strip().str.lower()
    print("World data for 2019 loaded:")
    print(data.head())
    return data

# Indlæs data og filtrer efter cluster
cluster_to_plot = 0  # Justér dette for at vælge et specifikt cluster
performance_data = load_performance_data(performance_file, cluster=cluster_to_plot)
world_data = load_world_data(world_data_file)

# Funktion til scatterplot

def plot_scatter(data, x_feature, y_feature, color_feature):
    if data.empty:
        print("Datasættet er tomt. Ingen punkter kan plottes.")
        return

    plt.figure(figsize=(12, 8))

    # Opret intervaller for farvegraduering
    bins = [-float("inf"), -25, -15, -5, 5, 15, 25, float("inf")]
    labels = ['<-25%', '-25% to -15%','-15 to -5', '-5% to 5%', '5% to 15%', '15% to 25%', '>25%']
    data['GB Fejl% Interval'] = pd.cut(data[color_feature], bins=bins, labels=labels)

    scatter = sns.scatterplot(
        data=data,
        x=x_feature,
        y=y_feature,
        hue='GB Fejl% Interval',
        palette='coolwarm',
        legend='full'
    )
    scatter.set_title(f'Scatterplot: {x_feature} vs {y_feature} (Farvekode: GB Fejl% Interval)', fontsize=16)
    scatter.set_xlabel(x_feature, fontsize=14)
    scatter.set_ylabel(y_feature, fontsize=14)
    '''
    # Tilføj landenavne ved hver prik
    for i in range(len(data)):
        plt.text(
            data[x_feature].iloc[i],
            data[y_feature].iloc[i],
            data['country'].iloc[i],
            fontsize=9,
            ha='right'
        )
    '''
    plt.legend(title='GB Fejl% Interval', fontsize=10, title_fontsize=12)
    plt.show()

# Eksempel på plot
# Justér her for at vælge features og generere scatterplot
selected_x = 'GB Fejl%'
selected_y = 'oil_consumption'
selected_color = 'GB Fejl%'

# Debugging: Tjek manglende lande
missing_in_world = set(performance_data['Land']) - set(world_data['country'])
missing_in_performance = set(world_data['country']) - set(performance_data['Land'])
print(f"Lande i performance_data men ikke i world_data: {missing_in_world}")
print(f"Lande i world_data men ikke i performance_data: {missing_in_performance}")

# Filtrer world_data for at matche lande fra performance_data
world_data_filtered = world_data[world_data['country'].isin(performance_data['Land'])]

# Fjern dubletter og behold kun én post per land
world_data_filtered = world_data_filtered.drop_duplicates(subset=['country'])

# Tilføj `GB Fejl%` fra performance_data til world_data_filtered
world_data_filtered = world_data_filtered.merge(
    performance_data[['Land', 'GB Fejl%']].rename(columns={'Land': 'country'}),
    on='country',
    how='left'
)

# Debugging: Tjek data efter merge
print("Filtered data:")
print(world_data_filtered.head())

# Generer scatterplot
plot_scatter(world_data_filtered, selected_x, selected_y, selected_color)
