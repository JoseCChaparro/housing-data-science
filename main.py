import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

import numpy as np

housing = load_housing_data().drop("median_house_value",axis=1)

housing_num = housing.drop("ocean_proximity", axis=1)

#PREPARANDO LOS DATOS
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

import numpy as np
import streamlit as st
import pandas as pd

# Definir la interfaz de usuario en Streamlit
st.title('''Housing Price Prediction App''')
st.markdown("""## José Carlos Chaparro Morales""")

median_income = st.number_input('Ingresar el ingreso medio', min_value=1)
total_rooms = st.number_input('Ingresar el total de habitaciones', min_value=1)
housing_median_age = st.number_input('Ingresar la edad media de las viviendas', min_value=1)
households =  st.number_input('Ingresar el total de hogares', min_value=1)
total_bedrooms = st.number_input('Ingresar el total de cuartos', min_value=1)
population = st.number_input('Ingresar la población total', min_value=1)
longitude = st.number_input('Ingresar la longitud')
latitude = st.number_input('Ingresar la latitud')
ocean_proximity = st.selectbox('Proximidad al océano', housing['ocean_proximity'].unique())

import joblib
def predict(data):
    lin_reg = joblib.load("forest_reg.pkl")
    return lin_reg.predict(data)


if st.button("Prediccion con Random Forest Regressor"):
    # Cargar los datos
    data = {'median_income': median_income,
        'total_rooms': total_rooms,
        'housing_median_age': housing_median_age,
        'households': households,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'longitude': longitude,
        'latitude': latitude,
        'ocean_proximity': ocean_proximity}

    features = pd.DataFrame(data, index=[0])

    data_prepared = full_pipeline.transform(features)

    result = predict(data_prepared)
    st.text(result[0])
