# ======== Konfigurationsvariabler ========
CONTINENT = 'Europe'  # Ændr dette til det ønskede kontinent, f.eks. 'Asia', 'Africa', etc.
DATA_FILE = f"{CONTINENT}_data.csv"
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"
FEATURES_SELECTED = "feature_1"
YEAR_SPLIT = 2009  # Årstal til at splitte data i trænings- og testdata
# =========================================