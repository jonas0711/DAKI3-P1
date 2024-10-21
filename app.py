import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ======== Konfigurationsvariabler ========
CONTINENT = 'Europe'  # Ændr dette til det ønskede kontinent, f.eks. 'Asia', 'Africa', etc.
DATA_FILE = f"{CONTINENT}_data.csv"
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"
FEATURES_FILENAME = f"features_list_{CONTINENT.lower()}.pkl"
TEST_YEAR_THRESHOLD = 2009  # Årstalsgrænse for testperiode
# =========================================

# Indlæs datasættet
df = pd.read_csv(DATA_FILE)

# Erstat '\n' og mellemrum med underscores i kolonnenavne for at matche modelens træning
df.columns = df.columns.str.replace('\n', ' ').str.replace(' ', '_')

# Indlæs den gemte model og features
try:
    features = joblib.load(FEATURES_FILENAME)
except FileNotFoundError:
    st.error(f"Feature-listen '{FEATURES_FILENAME}' blev ikke fundet. Kør 'features.py' først.")
    st.stop()

try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    st.error(f"Modellen '{MODEL_FILENAME}' blev ikke fundet. Kør 'features.py' først.")
    st.stop()

# Håndterer kategoriske data (country) via one-hot encoding som i træning
df_encoded = pd.get_dummies(df, columns=['country'])

# Begrænser data til testperioden (2011 og frem)
test_years = df[df['year'] > TEST_YEAR_THRESHOLD]['year'].unique()

# Normaliserer data som i træningen
scaler = StandardScaler()
# Vi antager, at feature-listen inkluderer alle nødvendige kolonner
try:
    scaler.fit(df_encoded[features].dropna())
except KeyError as e:
    st.error(f"En eller flere features mangler i datasættet: {e}")
    st.stop()

# Streamlit-app opsætning
st.title("Forudsigelse af CO₂-udledninger med Machine Learning")

st.write(
    """
    Dette værktøj forudsiger CO₂-udledninger for et valgt land og årstal ved hjælp af en trænet 
    maskinlæringsmodel. Modellen anvender data som bruttonationalprodukt (BNP), fossilt energiforbrug,
    vedvarende energiforbrug og befolkningstal. Du kan sammenligne modelens forudsigelse med de faktiske
    udledninger.
    """
)

# Land- og årstalvalg
# Sørg for kun at hente unikke lande fra 'country'-kolonnen, og ikke hele data-rækker
countries = df['country'].unique()  # Dette vil kun få landenes navne
selected_country = st.selectbox("Vælg et Land:", countries)

# Årstalsvalg
selected_year = st.selectbox("Vælg Årstal (kun testperiode):", sorted(test_years))

# Filtrerer data for det valgte land og år
encoded_country_col = f'country_{selected_country}'
if encoded_country_col not in df_encoded.columns:
    st.write(f"Data ikke tilgængelig for det valgte land: {selected_country}")
else:
    selected_data = df_encoded[(df_encoded[encoded_country_col] == 1) & (df_encoded['year'] == selected_year)]

    if not selected_data.empty:
        # Brug de samme features som modellen blev trænet med
        try:
            X_new = selected_data[features]
        except KeyError as e:
            st.error(f"En eller flere features mangler i det valgte datasæt: {e}")
            st.stop()

        # Hent den faktiske værdi
        actual_row = df[(df['country'] == selected_country) & (df['year'] == selected_year)]
        if actual_row.empty:
            st.write("Faktiske data ikke tilgængelige for det valgte land og år.")
        else:
            y_actual = actual_row['Value_co2_emissions_kt_by_country'].values[0]

            # Normaliserer de nye data
            X_new_normalized = scaler.transform(X_new)

            # Forudsiger med den gemte model
            y_pred = model.predict(X_new_normalized)[0]

            # Beregn forskellen og procentvis afvigelse
            difference = y_actual - y_pred
            percent_difference = (abs(difference) / y_actual) * 100 if y_actual != 0 else 0

            # Præsentation af forudsigelse og faktiske værdier
            st.subheader("Forudsigelse sammenlignet med Faktiske Data")
            st.write(f"**Faktisk CO₂-udledning:** {y_actual:,.2f} kt")
            st.write(f"**Forudsagt CO₂-udledning:** {y_pred:,.2f} kt")
            st.write(f"**Afvigelse:** {difference:,.2f} kt")
            st.write(f"**Procentvis Afvigelse:** {percent_difference:.2f}%")

            # Visualisering af forudsigelse og faktiske data
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(['Faktisk', 'Forudsagt'], [y_actual, y_pred], color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel("CO₂-udledninger (kt)")
            ax.set_title(f"CO₂-udledning for {selected_country} i {selected_year}")

            for i, v in enumerate([y_actual, y_pred]):
                ax.text(i, v + 0.05 * max(y_actual, y_pred), f"{v:,.2f} kt", ha='center', fontweight='bold')

            st.pyplot(fig)
    else:
        st.write("Data ikke tilgængelig for det valgte land og år.")
