import json
import pandas as pd
from glom import glom

df = pd.read_csv('C:/Users/86185/Desktop/myvar.csv')
print(df)
print(df['label'].isnull())