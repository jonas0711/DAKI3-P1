# CO2-Udledning Prognosemodel - En Guide for Begyndere

Dette projekt er udviklet til at forudsige CO2-udledninger på tværs af forskellige lande og kontinenter. Projektet bruger maskinlæringsmodeller til at analysere historiske data og lave fremtidige prognoser baseret på forskellige samfunds- og energifaktorer.

## Hvad Kan Programmet?

Programmet kan:
- Forudsige CO2-udledninger for forskellige lande og kontinenter
- Analysere sammenhænge mellem forskellige faktorer og CO2-udledning
- Sammenligne forskellige maskinlæringsmodellers præstationer
- Visualisere korrelationer mellem forskellige features
- Opdele og analysere data på kontinentbasis

## Hvordan Virker Det?

### Kontrolcenter - Vores "Kommandocentral"
I filen `kontrolcenter.py` har vi samlet de centrale konfigurationsvariabler:

```python
CONTINENT = 'world'  # Vælg mellem 'world' eller specifikt kontinent
DATA_FILE = f"CSV_files/{CONTINENT}_data.csv"  # Automatisk valg af datafil
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"  # Navnet på den gemte model
FEATURES_SELECTED = "features_psb"  # Hvilke features der skal bruges
YEAR_SPLIT = 2009  # Skilleår mellem trænings- og testdata
```

### Data Vi Bruger
Programmet analyserer en bred vifte af faktorer:
- Befolkningstæthed og befolkningstal
- BNP og økonomisk vækst
- Energiforbrug og -produktion
- Vedvarende energiandel
- Elektricitetsproduktion fra forskellige kilder
- Drivhusgasudledninger
- Geografisk placering (kontinent)

## Projektstruktur

```
Projektmappe/
├── CSV_files/               <- Datamapper med CSV-filer for hvert kontinent
├── kontrolcenter.py        <- Central konfigurationsfil
├── correlation.py          <- Analyserer korrelationer mellem features
├── features.py            <- Håndterer databehandling og feature selection
├── models.py              <- Implementerer forskellige ML-modeller
├── opdel_i_kontinenter.py <- Opdeler data efter kontinenter
└── udvalgte_features.json <- Definition af feature-sæt
```

## Modeller i Projektet

Projektet inkluderer flere forskellige maskinlæringsmodeller:
1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. Random Forest Regression
5. Gradient Boosting Regression
6. Support Vector Regression

## Sådan Kommer Du I Gang

### 1. Forberedelse
Du skal have følgende installeret:
- Python 3.x
- Nødvendige Python-pakker:
  - pandas (databehandling)
  - scikit-learn (maskinlæring)
  - numpy (numeriske beregninger)
  - joblib (gem/indlæs modeller)
  - matplotlib (visualisering)
  - seaborn (visualisering)

Installation af pakker:
```bash
pip install pandas scikit-learn numpy joblib matplotlib seaborn
```

### 2. Opsætning
1. Klon eller download projektet
2. Åbn `kontrolcenter.py`
3. Vælg ønsket kontinent og andre indstillinger

### 3. Kør Analyserne

For at køre korrelationsanalyse:
```bash
python correlation.py
```

For at træne og teste modeller:
```bash
python models.py
```

For at opdele data i kontinenter:
```bash
python opdel_i_kontinenter.py
```

## Arbejdsflow

1. **Dataopdeling** (`opdel_i_kontinenter.py`)
   - Opdeler det globale datasæt i kontinentspecifikke datasæt
   - Tilføjer kontinentinformation til dataene

2. **Feature Selection** (`features.py`)
   - Håndterer missing values
   - Normaliserer data
   - Udfører one-hot encoding på kategoriske variable
   
3. **Korrelationsanalyse** (`correlation.py`)
   - Visualiserer korrelationer mellem features
   - Identificerer de vigtigste faktorer for CO2-udledning

4. **Modeltræning** (`models.py`)
   - Træner forskellige modeller på historisk data
   - Evaluerer modellernes præstation
   - Gemmer den bedste model til senere brug

## Resultater og Evaluering

Programmet evaluerer modellerne på to måder:
- R² score (accuracy): Hvor god modellen er til at forklare variationen i data
- RMSE (Root Mean Squared Error): Den gennemsnitlige fejl i forudsigelserne

## Fejlfinding

Almindelige problemer og løsninger:
- "ModuleNotFoundError": Installer den manglende pakke med `pip install pakkenavn`
- "FileNotFoundError": Tjek at alle CSV-filer er i CSV_files mappen
- "KeyError": Kontroller at feature-navnene i udvalgte_features.json matcher dem i din CSV-fil

## Eksperimenter og Tilpasninger

Du kan eksperimentere med:
1. Forskellige feature-sæt i udvalgte_features.json
2. Forskellige maskinlæringsmodeller i models.py
3. Forskellige år for trænings/test-split
4. Andre parametre i modellerne

## Ordliste

- **Feature**: En specifik egenskab eller måling vi bruger til at lave forudsigelser
- **One-hot encoding**: Konvertering af kategoriske data til numerisk format
- **R² score**: Et mål for hvor god modellen er (0-1, hvor 1 er perfekt)
- **RMSE**: Et mål for den gennemsnitlige fejl i forudsigelserne
- **Normalisering**: Proces der gør forskellige features sammenlignelige
