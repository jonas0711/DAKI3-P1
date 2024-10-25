import pandas as pd

# Opret en dictionary for kontinenter som før
country_to_continent = {
    'Algeria': 'Africa', 'Argentina': 'South America', 'Australia': 'Oceania', 
    'Austria': 'Europe', 'Azerbaijan': 'Asia', 'Bangladesh': 'Asia', 
    'Belarus': 'Europe', 'Belgium': 'Europe', 'Brazil': 'South America', 
    'Bulgaria': 'Europe', 'Canada': 'North America', 'Chile': 'South America', 
    'China': 'Asia', 'Colombia': 'South America', 'Croatia': 'Europe', 
    'Cyprus': 'Europe', 'Czechia': 'Europe', 'Denmark': 'Europe', 
    'Ecuador': 'South America', 'Egypt': 'Africa', 'Estonia': 'Europe', 
    'Finland': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 
    'Greece': 'Europe', 'Hungary': 'Europe', 'Iceland': 'Europe', 
    'India': 'Asia', 'Indonesia': 'Asia', 'Iraq': 'Asia', 'Ireland': 'Europe', 
    'Israel': 'Asia', 'Italy': 'Europe', 'Japan': 'Asia', 'Kazakhstan': 'Asia', 
    'Kuwait': 'Asia', 'Latvia': 'Europe', 'Lithuania': 'Europe', 
    'Luxembourg': 'Europe', 'Malaysia': 'Asia', 'Mexico': 'North America', 
    'Morocco': 'Africa', 'Netherlands': 'Europe', 'New Zealand': 'Oceania', 
    'North Macedonia': 'Europe', 'Norway': 'Europe', 'Oman': 'Asia', 
    'Pakistan': 'Asia', 'Peru': 'South America', 'Philippines': 'Asia', 
    'Poland': 'Europe', 'Portugal': 'Europe', 'Qatar': 'Asia', 
    'Romania': 'Europe', 'Saudi Arabia': 'Asia', 'Singapore': 'Asia', 
    'Slovakia': 'Europe', 'Slovenia': 'Europe', 'South Africa': 'Africa', 
    'Spain': 'Europe', 'Sri Lanka': 'Asia', 'Sweden': 'Europe', 
    'Switzerland': 'Europe', 'Thailand': 'Asia', 'Trinidad and Tobago': 'North America', 
    'Turkey': 'Europe', 'Turkmenistan': 'Asia', 'Ukraine': 'Europe', 
    'United Arab Emirates': 'Asia', 'United Kingdom': 'Europe', 
    'United States': 'North America', 'Uzbekistan': 'Asia'
}

# Læs din CSV-fil
df = pd.read_csv(r'CSV_files/Energi_Data.csv')

# Gem en kopi af hele datasættet som world_data.csv
df.to_csv('world_data.csv', index=False)
print(f"Data for hele verden gemt i filen: world_data.csv")

# Tilføj en kolonne til DataFrame, der repræsenterer kontinenter
df['continent'] = df['country'].map(country_to_continent)

# Loop igennem hver unikke kontinent og gem data til en separat CSV-fil
for continent in df['continent'].unique():
    # Filtrer data for det pågældende kontinent
    df_continent = df[df['continent'] == continent]
    
    # Opret et filnavn baseret på kontinentets navn
    file_name = f"{continent}_data.csv"
    
    # Gem data som CSV-fil
    df_continent.to_csv(file_name, index=False)
    
    print(f"Data for {continent} gemt i filen: {file_name}")