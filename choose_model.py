import pandas as pd
from settings.constants import TRAIN_CSV, VAL_CSV
from utils.dataloader import DataLoader

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

train = pd.read_csv(TRAIN_CSV, header=0)
val = pd.read_csv(VAL_CSV, header=0)
full_data = [train, val]

X_raw = train.drop("price", axis=1)

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
y = train["price"]

# for test
X_raw_val = val.drop("price", axis=1)
loader.fit(X_raw_val)
X_val = loader.load_data()
y_val = val['price']

models = [
    LinearRegression(),
    RidgeCV(),
    LassoCV(),
    ElasticNet(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor()
]

log_cols = ["Predictor", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

kf = KFold(n_splits=10, random_state=42, shuffle=True)

acc_dict = {}

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for mdl in models:
        name = mdl.__class__.__name__
        mdl.fit(X_train, y_train)

        # y_pred = clf.predict(X_test)
        # acc = mean_absolute_percentage_error(y_test, y_pred)

        acc = mdl.score(X_test, y_test)

        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc


for mdl in acc_dict:
    acc_dict[mdl] = acc_dict[mdl] / 10.0
    log_entry = pd.DataFrame({"Predictor": mdl, "Accuracy": acc_dict[mdl]}, columns=log_cols, index=[mdl])
    log = pd.concat([log, log_entry], ignore_index=True)

plt.xlabel('Accuracy')
plt.title('Predictor Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Predictor', data=log, color="g")
print(log)
