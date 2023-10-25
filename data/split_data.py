import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('kc_house_data.csv')

train, val = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
