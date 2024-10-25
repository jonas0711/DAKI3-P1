from kontrolcenter import *

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json

dataset = pd.read_csv(DATA_FILE)

# Udvælger features
'''Selecting features in the dataset'''
if dataset is None:
    dataset = pd.read_csv(DATA_FILE)

    # Rens kolonnenavne for ekstra semikoloner og mellemrum
dataset.columns = dataset.columns.str.replace(';', '').str.strip()

    # Erstat '\n' og mellemrum med underscore i alle kolonnenavne
dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

with open('udvalgte_features.json', 'r') as file:
    feature_schema = json.load(file)

features = feature_schema[FEATURES_SELECTED]
features.remove('country')
##features.remove('continent')

target = "Value_co2_emissions_kt_by_country"

    ##print("\nDatasæt information:")
    ##print(f"Antal rækker: {len(dataset)}")
    ##print(f"Antal kolonner: {len(dataset.columns)}")
    
    # Analysér manglende data
    ##print("\nAnalyse af manglende data:")
    ##missing_info = analyze_missing_data(dataset)

    # Fjern rækker med manglende værdier for både features og target
end_data = dataset[features + [target]].dropna()


# correlation matrix to see if some features explain the same variance
plt.figure(figsize=(12,10))
sns.heatmap(end_data.corr(), annot=True, cmap="magma", fmt='.2f')
plt.title('Correlation map')
plt.show()

print(end_data.corr())

# Efter den eksisterende kode i correlation.py, tilføj:

# Beregn korrelation med target variabel
target_correlations = end_data.corr()[target].sort_values(ascending=False)

print("\nKorrelationer med CO2-udledning (sorteret):")
print(target_correlations)