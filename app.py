import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Indlæser datasættet med den korrekte filsti
df = pd.read_csv(r"C:\Users\jonas\Desktop\Design og anvendelse af kunstig inteligens\GitHub Repo\P1-DAKI3\DAKI3-P1\Europe_data.csv")
model = joblib.load('gradient_boosting_model_Europe_2000_2010.pkl')

# Definerer features og scaler for at normalisere dataene som i træning
features = ['gdp', 'fossil_fuel_consumption', 'renewables_consumption', 'population']
scaler = StandardScaler()
scaler.fit(df[features].dropna())

# Begrænser data til testperioden (2011 og frem)
test_years = df[(df['year'] > 2010)]['year'].unique()

# Opsætning af Streamlit-app
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
countries = df['country'].unique()
selected_country = st.selectbox("Vælg et Land:", countries)
selected_year = st.selectbox("Vælg Årstal (kun testperiode):", sorted(test_years))

# Filtrerer data for det valgte land og år
selected_data = df[(df['country'] == selected_country) & (df['year'] == selected_year)]

if not selected_data.empty:
    # Henter de nødvendige features fra datasættet
    X_new = selected_data[features]
    y_actual = selected_data['Value_co2_emissions_kt_by_country'].values[0]
    
    # Normaliserer de nye data
    X_new_normalized = scaler.transform(X_new)
    
    # Forudsiger med den gemte model
    y_pred = model.predict(X_new_normalized)[0]
    
    # Beregner forskellen og procentvis afvigelse
    difference = y_actual - y_pred
    percent_difference = (abs(difference) / y_actual) * 100
    
    # Præsentation af forudsigelse og faktiske værdier
    st.subheader("Forudsigelse sammenlignet med Faktiske Data")
    st.write(f"**Faktisk CO₂-udledning:** {y_actual:,.2f} kt")
    st.write(f"**Forudsagt CO₂-udledning:** {y_pred:,.2f} kt")
    st.write(f"**Afvigelse:** {difference:,.2f} kt")
    st.write(f"**Procentvis Afvigelse:** {percent_difference:.2f}%")
    
    # Visualisering af forudsigelse og faktiske data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Faktisk', 'Forudsagt'], [y_actual, y_pred], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel("CO₂-udledninger (kt)", fontsize=12)
    ax.set_title(f"CO₂-udledning for {selected_country} i {selected_year}", fontsize=14)
    
    # Tilføjer værdier til søjlediagrammet
    for i, v in enumerate([y_actual, y_pred]):
        ax.text(i, v + 0.05 * max(y_actual, y_pred), f"{v:,.2f} kt", ha='center', fontweight='bold')
    
    st.pyplot(fig)

else:
    st.write("Data ikke tilgængelig for det valgte land og år.")
