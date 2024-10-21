# Importerer nødvendige biblioteker
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Indlæser datasættet med den korrekte filsti
df = pd.read_csv(r"C:\Users\jonas\Desktop\Design og anvendelse af kunstig inteligens\GitHub Repo\P1-DAKI3\DAKI3-P1\Europe_data.csv")

# Definerer features og målvariabel
features = ['gdp', 'fossil_fuel_consumption', 'renewables_consumption', 'population']  # Opdater efter behov
target = 'Value_co2_emissions_kt_by_country'  # Målvariabel

# Filtrerer datasættet til de nødvendige kolonner og fjerner rækker med manglende værdier i disse kolonner
df_filtered = df[features + [target]].dropna()

# Opdel i X (features) og y (målvariabel)
X = df_filtered[features]
y = df_filtered[target]

# Sikrer, at alle kolonner er numeriske
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Normaliserer dataene
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Opdel datasættet i trænings- og test-sæt baseret på årstal (2000-2010 til træning, 2011 og frem til test)
train_data = df[(df['year'] <= 2010) & (df[features + [target]].notnull().all(axis=1))]
test_data = df[(df['year'] > 2010) & (df[features + [target]].notnull().all(axis=1))]

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Normaliserer både trænings- og testdata
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Initialisering af Gradient Boosting-modellen
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Træner modellen
model.fit(X_train_normalized, y_train)

# Evaluerer modellen på testdata
y_pred = model.predict(X_test_normalized)
print("\nEvaluering af Gradient Boosting-model:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Gemmer den trænede model til en fil
model_filename = 'gradient_boosting_model_Europe_2000_2010.pkl'
joblib.dump(model, model_filename)
print(f"Model gemt som {model_filename}")
