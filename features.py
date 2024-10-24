import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import json

# ======== Konfigurationsvariabler ========
CONTINENT = 'Europe'  # Ændr dette til det ønskede kontinent, f.eks. 'Asia', 'Africa', etc.
DATA_FILE = f"{CONTINENT}_data.csv"
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"
FEATURES_SELECTED = "feature_1"
YEAR_SPLIT = 2009  # Årstal til at splitte data i trænings- og testdata
# =========================================

# Indlæs dataset
dataset = pd.read_csv(DATA_FILE)

def features_handling(dataset, year=YEAR_SPLIT):
    # Erstat '\n' og mellemrum med underscore i alle kolonnenavne
    dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

    with open('udvalgte_features.json', 'r') as file:
        feature_schema = json.load(file)
    
    print(feature_schema)  # Udskriv hele ordbogen
    print(FEATURES_SELECTED)
    features = feature_schema[FEATURES_SELECTED]

    '''[
        "country",  # Sørg for at inkludere "country" kolonnen her, hvis den er en del af datasættet.
        "year", 
        "Density_(P/Km2)", 
        "gdp_growth", 
        "Access_to_electricity_(%_of_population)", 
        "Primary_energy_consumption_per_capita_(kWh/person)", 
        "Renewable_energy_share_in_the_total_final_energy_consumption_(%)", 
        "Low-carbon_electricity_(%_electricity)", 
        "fossil_cons_change_pct", 
        "low_carbon_cons_change_pct", 
        "coal_share_energy", 
        "energy_per_gdp", 
        "fossil_share_energy", 
        "gas_share_energy", 
        "hydro_share_energy", 
        "low_carbon_share_energy", 
        "nuclear_share_energy", 
        "oil_share_energy", 
        "other_renewables_share_energy", 
        "renewables_share_energy", 
        "solar_share_energy", 
        "coal_share_energy"
    ]'''

    target = "Value_co2_emissions_kt_by_country"

    # Fjern rækker med manglende værdier for både features og target
    end_data = dataset[features + [target]].dropna()

    # Kontroller om 'country'-kolonnen er til stede, før du udfører one-hot encoding
    if 'country' in end_data.columns:
        # Konverter kategoriske data til numeriske værdier (One-hot encoding for "country")
        end_data = pd.get_dummies(end_data, columns=["country"])
    else:
        print("Kolonnen 'country' findes ikke i datasættet.")

    # Gem de opdaterede features efter one-hot encoding (de oprindelige + one-hot encoded kolonner)
    final_features = end_data.drop(columns=[target]).columns.tolist()

    # Gem de anvendte feature-navne efter one-hot encoding
    joblib.dump(final_features, FEATURES_SELECTED)
    print(f"Features gemt som {FEATURES_SELECTED}")

    # Adskil features og target
    X = end_data.drop(columns=[target])
    y = end_data[target]

    # Split data i trænings- og testdata baseret på årstal
    train_data = end_data[(end_data['year'] <= year)]
    test_data = end_data[(end_data['year'] > year)]

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    # Normaliserer både trænings- og testdata
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Gradient Boosting Regressor model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Træner modellen
    model.fit(X_train_normalized, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test_normalized)

    # Beregn R² score (accuracy) og Root Mean Squared Error (RMSE)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² score (accuracy): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Gemmer den trænede model til en fil
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model gemt som {MODEL_FILENAME}")

def select_data():
    '''Selecting features in the dataset'''

def split_data():
    '''Splitting dataset into train and test sæt'''

def scaler_data():
    '''Scaling the dataset'''

# Kald funktionen og print resultatet
features_handling(dataset)
