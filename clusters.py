from kontrolcenter import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Indlæsning af fil
data = pd.read_csv(DATA_FILE)
data = data[["Country","Value_co2_emissions_kt_by_country"]]

# Splitting i træning og test
X_train, X_test = train_test_split(data, test_size=0.2, random_state=39)

# Inspicer plot se efter naturlige klyning
