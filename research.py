import pandas as pd

# Indsæt filstien til dit CSV-dataset
file_path = r'C:\Users\jonas\Desktop\Design og anvendelse af kunstig inteligens\GitHub Repo\P1-DAKI3\DAKI3-P1\Energi_Data.csv'

# Læs CSV-filen ind i en DataFrame
df = pd.read_csv(file_path)

# Vis unikke lande i dataset
unique_countries = df['country'].unique()

# Print listen over lande
print("Unikke lande i dataset:")
print(unique_countries)

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

# Tilføj en ny kolonne til DataFrame, der repræsenterer kontinenter
df['continent'] = df['country'].map(country_to_continent)

# Gruppér landene efter kontinenter og optæl antallet af lande i hvert kontinent
continent_counts = df['continent'].value_counts()

# Vis optælling for hvert kontinent
print("Antal lande per kontinent:")
print(continent_counts)

# Vis en liste af lande for hvert kontinent
print("\nLande i hvert kontinent:")
for continent in continent_counts.index:
    countries_in_continent = df[df['continent'] == continent]['country'].unique()
    print(f"{continent}: {len(countries_in_continent)} lande")
    print(", ".join(countries_in_continent))
    print()  # Tom linje for bedre læsbarhed
