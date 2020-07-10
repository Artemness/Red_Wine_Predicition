import pandas as pd
import numpy as np

df = pd.read_csv('winequality-red.csv')

#Rounding Chlorides and Density to 3 and 4 decimal places respectfully:
df['chlorides'] = df['chlorides'].apply(lambda x: round(x,3))
df['density'] = df['density'].apply(lambda x: round(x,4))

#Checking Data to see for missing values:
print(df.info())

df.to_csv('wineclean.csv', index= False)