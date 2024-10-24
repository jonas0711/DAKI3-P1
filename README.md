# CO2-Udledning Prognosemodel - En Guide for Begyndere

Dette projekt er lavet til at forudsige hvor meget CO2 forskellige lande udleder. Tænk på det som en avanceret lommeregner, der kan gætte fremtidige CO2-udledninger baseret på forskellige informationer om landet.

## Hvad Kan Programmet?

Programmet kan:
- Forudsige hvor meget CO2 et land vil udlede
- Lære fra historiske data (ligesom når du lærer af erfaring)
- Tage højde for mange forskellige faktorer som påvirker CO2-udledning

## Hvordan Virker Det?

### Variabler - Vores "Kontrolpanel"
I filen `globale_variabler.py` har vi samlet alle de indstillinger, som vi let kan ændre. Det fungerer som et kontrolpanel, hvor du kan:

```python
CONTINENT = 'Europe'  # Her vælger du hvilket kontinent du vil undersøge
DATA_FILE = f"CSV_files/{CONTINENT}_data.csv"  # Her finder programmet den rigtige datafil
YEAR_SPLIT = 2009  # Her bestemmer du, hvilke år der skal bruges til træning
```

Dette gør det nemt at:
- Skifte mellem forskellige kontinenter (f.eks. 'Europe', 'Asia', 'Africa')
- Bruge forskellige datasæt
- Ændre hvilke år vi træner modellen med

Det er ligesom at have forskellige opskrifter i en kogebog - du kan let ændre ingredienserne (data) og fremgangsmåden (årene), uden at skulle omskrive hele opskriften.

### Data Vi Bruger
Programmet kigger på mange forskellige ting for at gætte CO2-udledningen:
- Hvor mange mennesker der bor i landet (befolkningstæthed)
- Hvor rigt landet er (BNP-vækst)
- Hvor meget strøm landet bruger
- Hvor meget vedvarende energi landet bruger (som sol og vind)
- Og mange andre faktorer

## Mappestruktur - Hvor Ligger Hvad?

Tænk på projektets filer som forskellige værktøjer i en værktøjskasse:

```
Projektmappe/
├── CSV_files/               <- Her ligger vores data (som Excel-filer, bare i et andet format)
│   └── Europa_data.csv     <- Data for europæiske lande
├── features.py             <- Håndterer vores data (som en køkkenassistent der forbereder ingredienser)
├── models.py               <- Vores "hjerne" der lærer og forudsiger (som en kok der laver maden)
├── globale_variabler.py    <- Vores kontrolpanel (hvor vi justerer vores indstillinger)
└── udvalgte_features.json  <- Liste over hvilke informationer vi bruger (som en indkøbsliste)
```

## Sådan Kommer Du I Gang

### 1. Forberedelse
Først skal du have installeret nogle værktøjer på din computer:
- Python (tænk på det som køkkenet)
- Nogle ekstra Python-pakker (som ekstra køkkenudstyr):
  - pandas (til at håndtere data)
  - scikit-learn (til at lave forudsigelser)
  - numpy (til matematiske udregninger)
  - joblib (til at gemme vores trænet model)

### 2. Opsætning
1. Download alle filerne til din computer
2. Åbn `globale_variabler.py` i et program der kan redigere tekst
3. Her kan du ændre `CONTINENT` til det kontinent du vil undersøge

### 3. Kør Programmet
1. Åbn din computers terminal eller kommandoprompt
2. Gå til mappen med filerne
3. Skriv: `python models.py`

## Hvad Sker Der Bag Kulisserne?

1. **Dataforberedelse** (`features.py`)
   - Programmet læser data fra CSV-filen
   - Renser data for fejl og mangler
   - Forbereder data så modellen kan forstå det
   
2. **Træning** (`models.py`)
   - Modellen lærer fra de gamle data (før 2009)
   - Som at lære af historien for at blive bedre til at gætte fremtiden

3. **Test og Resultater**
   - Modellen prøver at gætte CO2-udledninger for nyere år (efter 2009)
   - Vi tjekker hvor god modellen er til at gætte rigtigt

## Resultater

Når programmet er færdigt, får du:
- En fil med den trænede model (som en erfaren kok der har lært opskriften)
- Information om hvor præcise forudsigelserne er
- Mulighed for at bruge modellen til at forudsige fremtidige CO2-udledninger

## Har Du Problemer?

Almindelige problemer og løsninger:
- Hvis programmet ikke kan finde din datafil: Tjek at den ligger i CSV_files mappen
- Hvis du får fejl om manglende pakker: Installer dem med `pip install pakkenavn`
- Hvis resultaterne ser mærkelige ud: Tjek at din data er i det rigtige format

## Vil Du Eksperimentere?

Prøv at:
1. Ændre `CONTINENT` til forskellige kontinenter
2. Justere `YEAR_SPLIT` til forskellige år
3. Se hvordan forskellige indstillinger påvirker resultaterne

Husk: Der er ingen fare ved at eksperimentere - programmet ændrer ikke i dine originale data!

## Begreber For Begyndere

- **CSV-fil**: En simpel måde at gemme data på (som et Excel-ark)
- **Model**: Et computerprogram der kan lære mønstre og lave forudsigelser
- **Træning**: Når modellen lærer fra eksisterende data
- **Variabler**: Værdier vi let kan ændre for at teste forskellige ting
- **Features**: De forskellige informationer vi bruger til at lave forudsigelser
