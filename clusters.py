from kontrolcenter import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

year = 2000

# Indlæsning af fil
data = pd.read_csv(DATA_FILE)
data = data[data['year'] == year]
data = data[["country","Value_co2_emissions_kt_by_country"]]

# Splitting i træning og test
X_train, X_test = train_test_split(data, test_size=0.2, random_state=39)

# Inspicer plot se efter naturlige klyning
plt.figure(figsize=(8, 6))
plt.scatter(data["county"],data["Value_co2_emissions_kt_by_country"],c='blue',label='Country clusters')
plt.title("Countries vs. CO2-emission pr. country")
plt.xlabel("Country")
plt.ylabel("CO2-emission (kt)")
plt.legend
plt.show

