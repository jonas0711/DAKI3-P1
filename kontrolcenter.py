# ======== Konfigurationsvariabler ========
DATA_FILE = f"CSV_files/world_data.csv"
MODEL_FILENAME = f"gradient_boosting_model_2000_2009.pkl"
FEATURES_SELECTED = "after_correlation"
YEAR_SPLIT = 2009  # Årstal til at splitte data i trænings- og testdata
TARGET = "Value_co2_emissions_kt_by_country"
TRAIN_YEARS = list(range(2000, 2010))  # Træningsår
TEST_YEARS = [2019]   # Kun test på 2019

# Cluster konfiguration
SELECTED_CLUSTER = "cluster_6"  # Vælg mellem: cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6
CLUSTERS_FILE = "clusters.json"
# =========================================