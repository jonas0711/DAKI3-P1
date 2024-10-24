# CO2 Emissions Prediction Model

This project implements a machine learning model to predict CO2 emissions by country using various energy and economic indicators. The model specifically focuses on continental regions and uses historical data to make predictions.

## Project Overview

The project uses a Gradient Boosting Regressor to predict CO2 emissions (kt) based on various features including energy consumption patterns, GDP growth, population density, and renewable energy adoption rates.

## Data Features

The model uses the following key features:
- Population density (P/Km²)
- GDP growth
- Access to electricity (% of population)
- Primary energy consumption per capita (kWh/person)
- Renewable energy share
- Low-carbon electricity percentage
- Various energy source distributions (coal, gas, hydro, nuclear, oil, solar, etc.)
- Fossil fuel consumption changes
- Energy efficiency metrics

## Project Structure

```
├── CSV_files/
│   └── {CONTINENT}_data.csv
├── features.py
├── models.py
├── globale_variabler.py
└── udvalgte_features.json
```

### Key Files

- `features.py`: Handles data preprocessing, feature selection, and scaling
- `models.py`: Contains the machine learning models (Random Forest and Gradient Boosting)
- `globale_variabler.py`: Contains global configuration variables
- `udvalgte_features.json`: Defines the feature sets used in the model

## Configuration

The project can be configured through `globale_variabler.py`:

```python
CONTINENT = 'Europe'  # Can be changed to other continents
DATA_FILE = f"CSV_files/{CONTINENT}_data.csv"
MODEL_FILENAME = f"gradient_boosting_model_{CONTINENT.lower()}_2000_2009.pkl"
FEATURES_SELECTED = "features_1"
YEAR_SPLIT = 2009  # Year to split training and test data
```

## Features and Data Processing

The data processing pipeline includes:
1. Feature selection from predefined sets
2. Handling missing values
3. One-hot encoding for categorical variables (countries)
4. Data splitting based on year
5. Feature scaling using StandardScaler

## Models

### Gradient Boosting Regressor
- Primary model used for predictions
- Configuration:
  - n_estimators: 100
  - learning_rate: 0.1
  - random_state: 42

### Random Forest Regressor
- Alternative model implementation available
- Used for comparison purposes

## Usage

1. Ensure your data is in the correct format and location (`CSV_files/{CONTINENT}_data.csv`)
2. Configure the desired continent and parameters in `globale_variabler.py`
3. Run the model:

```python
python models.py
```

The trained model will be saved as a `.pkl` file with the format `gradient_boosting_model_{continent}_2000_2009.pkl`

## Model Evaluation

The model performance is evaluated using:
- R² Score (accuracy)
- Root Mean Squared Error (RMSE)

## Data Split

- Training data: Years <= 2009
- Test data: Years > 2009

## Dependencies

- pandas
- scikit-learn
- numpy
- joblib

## File Format Requirements

The input CSV file should contain all the features listed in `udvalgte_features.json` plus a target column named "Value_co2_emissions_kt_by_country".

## Output

The model saves:
- Trained model as a pickle file
- Feature names used in the model
- Performance metrics (printed to console)
