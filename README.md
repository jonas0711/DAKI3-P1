# CO2-Udledning Prognosemodel - En Guide for Begyndere

Dette projekt er udviklet til at forudsige CO2-udledninger på tværs af forskellige lande og kontinenter. Tænk på det som en avanceret opskrift, hvor vi bruger forskellige ingredienser (data) til at lave en ret (forudsigelse) om fremtidig CO2-udledning.

## Hvad Kan Programmet?

Programmet kan:
- Forudsige CO2-udledninger for forskellige lande (som en kok der kan forudsige hvordan en ret vil smage)
- Analysere sammenhænge mellem forskellige faktorer (som at forstå hvordan ingredienserne påvirker hinanden)
- Sammenligne forskellige maskinlæringsmodeller (som at sammenligne forskellige tilberedningsmetoder)
- Visualisere korrelationer (som et smagskort over ingredienserne)
- Opdele og analysere data på kontinentbasis (som at tilpasse opskriften til forskellige køkkener)

## Hvordan Virker Det?

### Kontrolcenter - Vores "Køkken"
I filen `kontrolcenter.py` har vi samlet vores grundopskrift:

```python
CONTINENT = 'world'  # Hvilket køkken vil vi lave mad fra?
DATA_FILE = f"CSV_files/{CONTINENT}_data.csv"  # Vores kogebog
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"  # Gem vores færdige opskrift
FEATURES_SELECTED = "features_psb"  # Vores ingrediensliste
YEAR_SPLIT = 2009  # Hvornår vi skifter fra træning til test (som at øve opskriften)
```

Dette gør det nemt at:
- Skifte mellem forskellige kontinenter (som at skifte mellem forskellige køkkener)
- Bruge forskellige datasæt (som forskellige kogebøger)
- Ændre træningsperioden (som at justere tilberedningstiden)

### Data Vi Bruger - Vores Ingredienser
Vores program analyserer mange forskellige faktorer:
- Befolkningstæthed (som antallet af gæster)
- BNP og økonomisk vækst (som budgettet for måltidet)
- Energiforbrug (som energien brugt i køkkenet)
- Vedvarende energi (som bæredygtige ingredienser)
- Elektricitetsproduktion (som forskellige tilberedningsmetoder)
- Drivhusgasudledninger (som den endelige smag)
- Geografisk placering (som forskellige madkulturer)

## Projektstruktur - Vores Køkken

```
Projektmappe/
├── CSV_files/               <- Vores spisekammer med råvarer (data)
├── kontrolcenter.py        <- Vores køkkenplan (hovedindstillinger)
├── correlation.py          <- Vores smagstest (analyserer sammenhænge)
├── features.py            <- Vores køkkenassistent (forbereder ingredienserne)
├── models.py              <- Vores kokkemetoder (forskellige tilberedninger)
├── opdel_i_kontinenter.py <- Vores måde at organisere ingredienser på
└── udvalgte_features.json <- Vores indkøbsliste (hvilke ingredienser vi bruger)
```

## Modeller i Projektet - Vores Tilberedningsmetoder

Vi har forskellige måder at tilberede vores data på (som forskellige madlavningsteknikker):
1. Linear Regression (som at koge - den mest grundlæggende metode)
2. Lasso Regression (som at dampe - mere kontrolleret og præcis)
3. Ridge Regression (som at stege - god til at håndtere mange ingredienser)
4. Random Forest Regression (som en buffet af forskellige teknikker)
5. Gradient Boosting Regression (som en forfinet gourmetmetode)
6. Support Vector Regression (som molekylær gastronomi - avanceret og specialiseret)

## Sådan Kommer Du I Gang

### 1. Forberedelse - Klargør Dit Køkken
Du skal have følgende installeret:
- Python 3.x (dit grundlæggende køkkenudstyr)
- Nødvendige Python-pakker (som specialiserede køkkenredskaber):
  ```bash
  pip install pandas scikit-learn numpy joblib matplotlib seaborn
  ```
  - pandas (din køkkenvægt og målebæger)
  - scikit-learn (din kogebog med opskrifter)
  - numpy (dit præcisionsudstyr)
  - joblib (din madopbevaring)
  - matplotlib og seaborn (din præsentationsservice)

### 2. Opsætning - Forbered Dit Køkken
1. Download alle filerne (som at samle dine køkkenredskaber)
2. Åbn `kontrolcenter.py` (din hovedopskrift)
3. Vælg dit køkken (kontinent) og indstillinger

### 3. Start Madlavningen - Kør Programmet

For at analysere ingredienserne:
```bash
python correlation.py
```

For at afprøve opskrifterne:
```bash
python models.py
```

For at organisere dit spisekammer:
```bash
python opdel_i_kontinenter.py
```

## Arbejdsflow - Fra Råvarer til Færdig Ret

1. **Dataopdeling** (`opdel_i_kontinenter.py`)
   - Sorterer ingredienser efter oprindelse
   - Organiserer data systematisk
   - Tilføjer geografisk information

2. **Feature Selection** (`features.py`)
   - Forbehandler ingredienserne (renser data)
   - Måler og vejer (normaliserer)
   - Kategoriserer ingredienserne (one-hot encoding)
   
3. **Korrelationsanalyse** (`correlation.py`)
   - Undersøger hvilke ingredienser der komplimenterer hinanden
   - Laver et smagskort (korrelationsmatrix)
   - Identificerer hovedingredienserne

4. **Modeltræning** (`models.py`)
   - Eksperimenterer med forskellige tilberedningsmetoder
   - Evaluerer resultaterne
   - Gemmer de bedste opskrifter

## Resultater og Evaluering - Smagstesten

Vi bedømmer vores "retter" på to måder:
- R² score (0-1): Som en restaurantanmeldelse, hvor 1 er en michelin-stjerne
- RMSE: Som afvigelsen fra den perfekte smag, jo mindre jo bedre

## Fejlfinding - Når Noget Går Galt i Køkkenet

Almindelige problemer og løsninger:
- "ModuleNotFoundError": Du mangler et køkkenredskab - installer det med pip
- "FileNotFoundError": Kan ikke finde ingredienserne - tjek CSV_files mappen
- "KeyError": Bruger en ingrediens der ikke er på listen - tjek feature-listen

## Eksperimenter - Leg i Køkkenet

Du kan eksperimentere med:
1. Forskellige ingredienskombinationer (feature-sæt)
2. Nye tilberedningsmetoder (modeller)
3. Ændre tilberedningstiden (træningsperiode)
4. Justere krydderingen (model-parametre)

## Ordliste - Køkkenordbogen

- **Feature**: En ingrediens i vores opskrift
- **One-hot encoding**: At forberede ingredienser på en særlig måde
- **R² score**: Vores smagskarakter (0-1)
- **RMSE**: Hvor præcis vores tilberedning er
- **Normalisering**: At sikre alle ingredienser er i de rette mængder

## Særlige Noter

- Der er ingen fare ved at eksperimentere - originale data forbliver urørte
- Start med simple opskrifter (modeller) før du prøver de avancerede
- Dokumentér dine eksperimenter - som at skrive noter i en kogebog
- Del dine resultater - som at dele en god opskrift

## Vil Du Vide Mere?

Prøv at:
1. Eksperimentere med forskellige kontinenter
2. Sammenligne forskellige modellers resultater
3. Visualisere dine resultater på nye måder
4. Tilføje nye features til analysen

Husk: Den bedste kok er den, der ikke er bange for at eksperimentere og lære af sine fejl!
