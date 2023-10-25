import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader(object):

    def fit(self, dataset):
        self.dataset = dataset.copy()

    # apply regex
    def get_title(self, name):
        pattern = ' ([A-Za-z]+)\.'
        title_search = re.search(pattern, name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    def load_data(self):

        # drop columns
        drop_elements = ["id",
                         "date",
                         "waterfront",
                         "view",
                         "zipcode",
                         "yr_renovated",
                         "sqft_living15",
                         "sqft_lot15"]

        self.dataset = self.dataset.drop(drop_elements, axis=1)

        self.dataset.fillna(0, inplace=True)

        # encode labels
        le = LabelEncoder()

        for column in ["bedrooms",
                       "bathrooms",
                       "sqft_living",
                       "sqft_lot",
                       "floors",
                       "condition",
                       "grade",
                       "sqft_above",
                       "sqft_basement",
                       "yr_built",
                       "lat",
                       "long"]:
            self.dataset[column] = le.fit_transform(self.dataset[column])

        # standartization
        scaler = StandardScaler()
        for column in ["bedrooms",
                       "bathrooms",
                       "sqft_living",
                       "sqft_lot",
                       "floors",
                       "condition",
                       "grade",
                       "sqft_above",
                       "sqft_basement",
                       "yr_built",
                       "lat",
                       "long"]:
            self.dataset[column] = scaler.fit_transform(self.dataset[column].values.reshape(-1, 1))

        return self.dataset