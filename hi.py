import pandas as pd

# Indlæs dataset
dataset = pd.read_csv(r'Energi_Data.csv')


    # Erstat '\n' og mellemrum med underscore i alle kolonnenavne
dataset.columns = dataset.columns.str.replace('\n', ' ').str.replace(' ', '_')

    # Loop gennem alle kolonnenavne og udskriv dem én efter én
for column in dataset.columns:
    print(column)