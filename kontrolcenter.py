# ======== Konfigurationsvariabler ========
CONTINENT = 'world'  # Ændr dette til det ønskede kontinent
DATA_FILE = f"CSV_files/{CONTINENT}_data.csv"
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"
FEATURES_SELECTED = "after_correlation"
YEAR_SPLIT = 2009  # Årstal til at splitte data i trænings- og testdata
TARGET = "Value_co2_emissions_kt_by_country"
TRAIN_YEARS = list(range(2000, 2010))  # Træningsår
TEST_YEARS = list(range(2010, 2019))   # Testår

# Cluster konfiguration
SELECTED_CLUSTER = "cluster_0"  # Vælg mellem: cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5
CLUSTERS_FILE = "clusters.json"
# =========================================