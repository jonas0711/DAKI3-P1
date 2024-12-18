# CO2-Udledning Prognosemodel

Dette projekt er udviklet som del af et P1-projekt på uddannelsen Design og Anvendelse af Kunstig Intelligens ved Aalborg Universitet. Formålet er at analysere og forudsige CO2-udledninger på tværs af forskellige lande og landegrupper ved hjælp af maskinlæringsmodeller.

## Projektets Hovedfunktioner

- Forudsigelse af CO2-udledninger for forskellige lande og landegrupper
- Avanceret clustering-analyse til gruppering af lande
- Sammenligning af forskellige maskinlæringsmodeller (GB, RF, SVR)
- Detaljeret korrelationsanalyse
- Visualisering af resultater og mønstre
- Håndtering af forskellige datasæt og landegrupperinger

## Projektstruktur

```
Projektmappe/
├── CSV_files/               # Datamappe med CSV-filer
├── Billeder/               # Gemte visualiseringer
├── Modeller/              # Gemte ML-modeller
├── Performance_png/       # Performance visualiseringer
├── kontrolcenter.py       # Central konfiguration
├── features.py           # Feature håndtering og databehandling
├── models.py             # ML-modeller implementering
├── correlation.py        # Korrelationsanalyse
├── plot_predictions.py   # Visualisering af forudsigelser
├── cluster.py           # Clustering implementering
├── README.md            # Denne fil
├── clusters.json        # Konfiguration af landegruppering
├── valid_countries.json # Liste over gyldige lande
└── udvalgte_features.json # Feature konfiguration
```

## Central Styring via kontrolcenter.py

Projektet styres centralt gennem `kontrolcenter.py`, hvor alle vigtige konfigurationer samles:

```python
DATA_FILE = "CSV_files/world_data.csv"
FEATURES_SELECTED = "after_correlation"
YEAR_SPLIT = 2009
TARGET = "Value_co2_emissions_kt_by_country"
TRAIN_YEARS = list(range(2000, 2010))
TEST_YEARS = [2019]
SELECTED_CLUSTER = "cluster_6"
```

## Implementerede Modeller

1. **Gradient Boosting (GB)**
   - Bedst præsterende model på tværs af clusters
   - God til mellemstore landes udledninger
   - Udfordret ved ekstreme værdier (Kina, Indien)

2. **Random Forest (RF)**
   - Stabil performance
   - Mindre udsving i forudsigelser
   - Lignende udfordringer som GB

3. **Support Vector Regression (SVR)**
   - Fungerer bedst på normaliserede data
   - Mindre præcis end GB og RF
   - Kræver mere computerkraft

## Clustering Analyse

Projektet bruger K-means clustering til at gruppere lande i 6 hovedkategorier:

1. Cluster 0: Mindre lande med moderat energiforbrug
2. Cluster 1: Kina
3. Cluster 2: USA
4. Cluster 3: Store industrialiserede økonomier
5. Cluster 4: Indien
6. Cluster 5: Store lande med høj vedvarende energi
7. Cluster 6: Alle lande kombineret

## Installation og Opsætning

1. **Påkrævede Python-pakker:**
```bash
pip install pandas scikit-learn numpy joblib matplotlib seaborn
```

2. **Klargøring af data:**
- Placer CSV-filer i `CSV_files` mappen
- Verificer gyldige lande i `valid_countries.json`
- Tjek feature konfiguration i `udvalgte_features.json`

3. **Konfiguration:**
- Åbn `kontrolcenter.py`
- Juster parametre efter behov
- Vælg ønsket cluster via `SELECTED_CLUSTER`

## Anvendelse

1. **Korrelationsanalyse:**
```bash
python correlation.py
```

2. **Model træning:**
```bash
python models.py
```

3. **Visualisering af resultater:**
```bash
python plot_predictions.py
```

## Databehandling

Projektet følger en struktureret tilgang til databehandling:

1. **Datarensning** (`features.py`)
   - Fjernelse af ugyldige lande
   - Håndtering af manglende værdier
   - Feature normalisering

2. **Feature Selection** (`udvalgte_features.json`)
   - Prædefinerede feature sæt
   - Optimerede feature kombinationer
   - Korrelationsbaseret udvælgelse

## Evaluering

Modellerne evalueres på følgende metrikker:
- R² score (forklaringsgrad)
- RMSE (root mean squared error)
- Procentvis afvigelse
- Fejlprocent per land

## Fejlfinding

Almindelige problemer og løsninger:
- **FileNotFoundError**: Tjek at alle CSV-filer er korrekt placeret
- **KeyError**: Verificer feature navne i konfigurationen
- **ValueError**: Tjek datatyper og manglende værdier

## Vedligeholdelse

For at vedligeholde og opdatere projektet:
1. Brug `kontrolcenter.py` til konfigurationsændringer
2. Dokumentér alle ændringer
3. Test nye features isoleret
4. Opdater konfigurationsfiler ved behov