import pickle
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from utils.dataloader import DataLoader
from settings.constants import TRAIN_CSV, VAL_CSV


with open('settings/specifications.json') as f:
    specifications = json.load(f)

raw_train = pd.read_csv(TRAIN_CSV)
x_columns = specifications['description']['X']
y_column = specifications['description']['y']

X_raw = raw_train[x_columns]

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
y = raw_train['price']


model = RandomForestRegressor()
model.fit(X, y)

with open('models/RandomForest.pickle', 'wb')as f:
    pickle.dump(model, f)
