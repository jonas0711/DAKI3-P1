# CO2-Udledning Prognosemodel - En Begynderguide

Dette projekt er udviklet til at analysere og forudsige CO2-udledninger på tværs af forskellige lande og grupper. Tænk på det som en avanceret opskrift, hvor vi bruger forskellige ingredienser (data) til at lave en ret (forudsigelse) om fremtidig CO2-udledning.

## Hvad Kan Programmet?

Programmet kan:
- Forudsige CO2-udledninger for forskellige lande og landegrupper
- Analysere mønstre og sammenhænge mellem forskellige faktorer
- Udføre clustering-analyser for at gruppere lande efter udviklingsprofiler
- Sammenligne forskellige maskinlæringsmodeller
- Visualisere korrelationer og udviklingsmønstre
- Analysere data på tværs af forskellige landegrupperinger

## Hvordan Virker Det?

### Kontrolcenter - Vores "Køkken"
I filen `kontrolcenter.py` har vi samlet vores grundopskrift:

```python
CONTINENT = 'world'  # Hvilket datasæt vil vi arbejde med?
DATA_FILE = f"CSV_files/{CONTINENT}_data.csv"  # Vores datakilde
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"  # Gem vores model
FEATURES_SELECTED = "features_1"  # Vores variabelliste
YEAR_SPLIT = 2009  # Skilleår mellem træning og test
TARGET = "Value_co2_emissions_kt_by_country"  # Vores målvariabel
```

Dette gør det nemt at:
- Skifte mellem forskellige landegrupper
- Anvende forskellige datasæt
- Justere træningsperioden
- Tilpasse modellen til forskellige formål

### Data Vi Bruger - Vores Ingredienser
Vi analyserer mange forskellige faktorer, herunder:
- Befolkningstæthed
- BNP og økonomisk vækst
- Energiforbrug per capita
- Andel af vedvarende energi
- Elektricitetsproduktion og -forbrug
- CO2-udledninger
- Fossilt brændstofforbrug
- Geografisk placering

## Projektstruktur - Vores Køkken

```
Projektmappe/
├── CSV_files/               <- Vores datamappe
├── kontrolcenter.py        <- Hovedkonfiguration
├── correlation.py          <- Korrelationsanalyse
├── features.py            <- Feature-håndtering
├── models.py              <- Maskinlæringsmodeller
├── claude_cluster.py      <- K-means clustering
├── claude_dbscan.py       <- DBSCAN clustering
├── claude_time_cluster.py <- Tidsbaseret clustering
├── opdel_i_kontinenter.py <- Landegruppering
└── udvalgte_features.json <- Feature-konfiguration
```

## Clustering-metoder - Vores Analyseredskaber

Vi bruger tre forskellige clustering-tilgange:
1. K-means Clustering (claude_cluster.py)
   - Grupperer lande baseret på lignende karakteristika
   - Bruger silhouette score til optimal gruppering
   
2. DBSCAN Clustering (claude_dbscan.py)
   - Finder naturlige grupperinger i data
   - Håndterer støj og outliers effektivt

3. Tidsbaseret Clustering (claude_time_cluster.py)
   - Analyserer udvikling over tid
   - Identificerer fælles udviklingsmønstre

## Maskinlæringsmodeller - Vores Værktøjer

Vi har implementeret følgende modeller:
1. Linear Regression (basal lineær model)
2. Lasso Regression (lineær model med regularisering)
3. Ridge Regression (alternativ regulariseringsmetode)
4. Random Forest Regression (ensemble-metode)
5. Gradient Boosting Regression (avanceret ensemble-metode)
6. Support Vector Regression (ikke-lineær modellering)

## Sådan Kommer Du I Gang

### 1. Installation af Nødvendige Pakker
```bash
pip install pandas scikit-learn numpy joblib matplotlib seaborn
```

### 2. Opsætning
1. Klon projektet
2. Åbn `kontrolcenter.py`
3. Vælg ønsket landegruppering og konfiguration

### 3. Kør Analyserne

For korrelationsanalyse:
```bash
python correlation.py
```

For clustering-analyse:
```bash
python claude_cluster.py
python claude_dbscan.py
python claude_time_cluster.py
```

For modeltræning:
```bash
python models.py
```

## Databehandling og Analyse

1. **Feature Selection** (`features.py`)
   - Datarensning og -forberedelse
   - Normalisering
   - Kategorisk encoding
   
2. **Korrelationsanalyse** (`correlation.py`)
   - Identificerer variable sammenhænge
   - Skaber korrelationsmatrix
   - Finder nøglevariable

3. **Clustering** (claude_*.py filer)
   - Grupperer lande efter forskellige metoder
   - Analyserer udviklingsmønstre
   - Identificerer lignende lande

4. **Modeltræning** (`models.py`)
   - Træner forskellige modeller
   - Evaluerer præstationer
   - Gemmer bedste modeller

## Evaluering og Resultater

Vi evaluerer vores modeller på:
- R² score (0-1): Forklaringsgrad
- RMSE: Gennemsnitlig afvigelse
- Silhouette score: Kvalitet af clustering

## Fejlfinding

Almindelige problemer og løsninger:
- "ModuleNotFoundError": Installer manglende pakker
- "FileNotFoundError": Tjek stien til CSV-filer
- "KeyError": Verificer feature-navne i konfigurationen


## Særlige Noter

- Brug `kontrolcenter.py` til at styre analyserne
- Dokumentér ændringer og resultater
- Vær opmærksom på datakvalitet og manglende værdier
- Brug clustering-resultaterne til at forstå landemønstre

## Vil Du Vide Mere?

Prøv at:
1. Eksperimentere med forskellige clustering-parametre
2. Sammenligne modellers resultater på tværs af landegrupper
3. Analysere tidsmønstre i data
4. Tilføje nye variable til analysen

God fornøjelse med projektet!
