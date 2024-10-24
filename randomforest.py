import features
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

def main():
    # Hent datasæt
    data = pd.read_csv(r'Energi_Data.csv')

    # Udvælger features
    selected_data = features.select_data(data)

    # Opdeling i train og test data
    X_train, y_train = features.split_data(selected_data)

    # Scalering af data --> Behøver ikke at være scaleret
    #X_train = features.scaler_data()

    # Definerer modellen & træning
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Tester på træningsdataen
    y_predict_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_predict_train)

    print(f'Random forest train accuracy: {round(acc_train*100,2)}')

if __name__ == "__main__":
    main()
