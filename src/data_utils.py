
# Standard library imports
import os
import pickle
from typing import Tuple

# Third party imports
from datetime import datetime, date
import numpy as np

# libraries for data manipulation
import pandas as pd
import path

# tools from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def save_data_checkpoint(data, path):
    # Obtiene la ruta completa
    complete_path = os.path.join(os.getcwd(), path)

    # Crea el directorio si no existe
    os.makedirs(os.path.dirname(complete_path), exist_ok=True)

    # Guarda el objeto 'data' en el archivo
    with open(complete_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return print(
        f"Object saved successfully in {complete_path}.")

def load_data_checkpoint(path):
    
    complete_path = os.path.join(os.getcwd(), path)

    if os.path.exists(complete_path):
        with open(complete_path, "rb") as f:
            filename = pickle.load(f)
    return filename

def get_features(train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
   
    X_train = train_df.drop('TARGET_LABEL_BAD=1', axis=1)
    y_train = train_df['TARGET_LABEL_BAD=1']
    X_test = test_df.drop('TARGET_LABEL_BAD=1', axis=1)
    y_test = test_df['TARGET_LABEL_BAD=1']

    return X_train, y_train, X_test, y_test

def get_train_val(X_train: pd.DataFrame, y_train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      shuffle=True)
     
    return X_train, X_val, y_train, y_val

def load_dataset(
 path_rel_dataset = "dataset/raw/PAKDD2010_Modeling_Data.txt" ,
 path_rel_variables = "dataset/raw/PAKDD2010_VariablesList.XLS"):


    header_path = os.path.join(os.getcwd(), path_rel_variables)
    dataset_path = os.path.join(os.getcwd(), path_rel_dataset)

    # take headers from VariablesList.XLS
    headers_df = pd.read_excel(header_path    )
    headers = headers_df["Var_Title"].tolist()

    # replace duplicate headers
    headers_unique = []
    for header in headers:
        if header not in headers_unique:
            headers_unique.append(header)
        else:
            header_duplicated = header + "_2"
            headers_unique.append(header_duplicated)

    # create train dataframe with headers
    dataset = pd.read_csv(
        dataset_path,
        sep="\t",
        encoding="unicode_escape",
        low_memory=False,
        index_col="ID_CLIENT",
        names=headers_unique,
    )

    return dataset

def clean_dataset(df):

    cat_features = df.select_dtypes("O").nunique()
    cat_num_features = df.select_dtypes("number").nunique()[
        df.select_dtypes("number").nunique() < 15
    ]

    # create lists with  constant classes, features with +50% empty or NaN values
    cat_constant = list(cat_features[cat_features == 1].index)
    num_constant = ["MONTHS_IN_THE_JOB"]
    catnum_constant = list(cat_num_features[cat_num_features == 1].index)
    catnum_constant_dist = [
        "POSTAL_ADDRESS_TYPE",
        "FLAG_DINERS",
        "FLAG_AMERICAN_EXPRESS",
        "FLAG_OTHER_CARDS",
    ]

    # calculte the half of the dataset
    half_percent_dataset = df.shape[0]/2

    empty_features = list(df.eq(" ").sum()[df.eq(" ").sum() > half_percent_dataset].index)
    nan_features = list(df.isna().sum()[df.isna().sum() > half_percent_dataset].index)

    # grouping lists in only one
    remove_features = [
        *nan_features,
        *empty_features,
        *catnum_constant,
        *catnum_constant_dist,
        *num_constant,
        *cat_constant,
    ]

    # save to pickles 
    path = os.path.join(os.getcwd(), "remove_features.pkl")

    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(remove_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # remove selected features from dataframes
    df = df.drop(remove_features, axis=1)
    return df


def split(df, label):


    X_train, X_test, y_train, y_test = train_test_split(
        df, df[label], test_size=0.20, random_state=42, stratify=df[label]
    )
    X_train = X_train.drop(label, axis=1)
    X_test = X_test.drop(label, axis=1)

    return X_train, X_test, y_train, y_test

def outliers(df):
    df.loc[df["OTHER_INCOMES"] > 190000.0, "OTHER_INCOMES"] = np.NaN
    df.loc[df["QUANT_DEPENDANTS"] > 20, "QUANT_DEPENDANTS"] = np.NaN
    df.loc[df["AGE"] < 17, "AGE"] = np.NaN

    # adapt feature classes to PAKDD2010_VariablesList.XLS
    df.loc[df["SEX"] == "N", "SEX"] = np.NaN
    df.loc[df["STATE_OF_BIRTH"] == "XX", "STATE_OF_BIRTH"] = np.NaN
    df.loc[df["MARITAL_STATUS"] == 0, "MARITAL_STATUS"] = np.NaN
    df.loc[df["RESIDENCE_TYPE"] == 0.0, "RESIDENCE_TYPE"] = np.NaN
    df.loc[df["OCCUPATION_TYPE"] == 0.0, "OCCUPATION_TYPE"] = np.NaN

    return df   

def impute_nan(train, test):

    # Replace empty string values with np.NAN.
    train = train.replace(" ", np.NaN)
    test = test.replace(" ", np.NaN)

    # create a dataframe with sum of NaN values of numerical features
    missing_num_col = pd.DataFrame(
        train.select_dtypes(include="number").isna().sum(), columns=["NaN"]
    )
    # filter dataframe by feature's name with Nan values != 0
    missing_num_col = missing_num_col["NaN"][
        (missing_num_col["NaN"] != 0)
    ].index

    # input median values for all numerical columns with missing data
    for feature in missing_num_col:
        train[feature] = train[feature].fillna(train[feature].median())
        test[feature] = test[feature].fillna(test[feature].median())

    # create a dataframe with sum of NaN values of non numerical features
    missing_non_numerical = pd.DataFrame(
        train.select_dtypes(include="O").isna().sum(), columns=["NaN"]
    )
    # filter dataframe by feature's name with Nan values != 0
    missing_non_numerical = missing_non_numerical["NaN"][
        (missing_non_numerical["NaN"] != 0)
    ].index

    # input mode values for all non numerical columns with missing data
    for feature in missing_non_numerical:
        train[feature] = train[feature].fillna(train[feature].mode().loc[0])
        test[feature] = test[feature].fillna(test[feature].mode().loc[0])

    return train, test    


def scaling(train, test):

    # create a list of features to scale without categorical numerical features
    cat_num_features = train.select_dtypes("number").nunique()[
        train.select_dtypes("number").nunique() < 15
    ]
    number_dtype = train.select_dtypes("number").columns
    num_features_list = list(set(number_dtype) - set(cat_num_features.index))
    # num_features_list.remove("PROFESSION_CODE")

    # instance scaler
    scaler = StandardScaler()

    # scale column by column with each corresponding fit between train and test
    for feature in num_features_list:
        train[feature] = scaler.fit_transform(train[[feature]])
        test[feature] = scaler.transform(test[[feature]])


    return train, test
  

    # instance scaler
    scaler = StandardScaler()

    # scale column by column with each corresponding fit between train and test
    for feature in num_features_list:
        train[feature] = scaler.fit_transform(train[[feature]])
        test[feature] = scaler.transform(test[[feature]])

    path = os.path.join(os.getcwd(), "data/raw/scaler.pkl")

    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(path, f, protocol=pickle.HIGHEST_PROTOCOL)
    return train, test
  

def encode(train, test):

    # categorical numerical list
    cat_num_features = train.select_dtypes("number").nunique()[
        train.select_dtypes("number").nunique() < 15
    ]
    cat_num_features_list = list(cat_num_features.index)
    cat_num_features_list.append("PROFESSION_CODE")
    cat_num_features_list.remove("PAYMENT_DAY_5")
    cat_num_features_list.remove("PAYMENT_DAY_10")
    cat_num_features_list.remove("PAYMENT_DAY_15")
    cat_num_features_list.remove("PAYMENT_DAY_20")
    cat_num_features_list.remove("PAYMENT_DAY_25")

    # categorical features list
    cat_features = train.select_dtypes("O").nunique()
    cat_features_list = list(cat_features.index)

    # total categorical features
    encoding_features = [*cat_num_features_list, *cat_features_list]

    # save to pickles 
    path = os.path.join(os.getcwd(), "notebook/encoding_features.pkl")

    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(encoding_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # remove features with lot of classes to perform ohe
    encoding_features.remove("CITY_OF_BIRTH")
    encoding_features.remove("RESIDENCIAL_CITY")
    encoding_features.remove("RESIDENCIAL_BOROUGH")

    # declare encoder with drop first
    ohe = OneHotEncoder(
        dtype=int, drop="first", sparse_output=False, handle_unknown="ignore"
    )
    ohe_fitted = ohe.fit(train[encoding_features])

    path = os.path.join(os.getcwd(), "notebook/ohe_fitted.pkl")

    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(ohe_fitted, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_encoded_dfs = []
    test_encoded_dfs = []

    # encoding categorical columns on train and test split
    for feature in encoding_features:
        ohe_train = ohe.fit_transform(train[[feature]])
        ohe_test = ohe.transform(test[[feature]])
        # create columns with categories minus first on train and test
        categories = list(ohe.categories_[0][1:])
        categories_list = [feature + str(s) for s in categories]

        # Agregar los nuevos DataFrames codificados a las listas
        # Pasar los índices originales al DataFrame
        train_encoded_dfs.append(pd.DataFrame(ohe_train, columns=categories_list, index=train.index))
        test_encoded_dfs.append(pd.DataFrame(ohe_test, columns=categories_list, index=test.index))

      # drop original column on train and test
        train.drop(columns=feature, inplace=True)
        test.drop(columns=feature, inplace=True)

    # Guardar los DataFrames codificados en un archivo pickle	

    path = os.path.join(os.getcwd(), "notebook/train_encoded_dfs.pkl")

    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(train_encoded_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    test.drop(columns=['CITY_OF_BIRTH', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH'], inplace=True)
    train.drop(columns=['CITY_OF_BIRTH', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH'], inplace=True)

    
    # Concatenar todos los DataFrames codificados de una vez
    train = pd.concat([train, *train_encoded_dfs], axis=1)
    test = pd.concat([test, *test_encoded_dfs], axis=1)

    return train, test
def convert_age(df):
    # Define los rangos de edades que quieres usar
    age_ranges = [(0, 18), (19, 30), (31, 50), (51, 70), (71, 100)]
    
    for i, age_range in enumerate(age_ranges):
        min_age, max_age = age_range
        # Crea una nueva columna para cada rango de edad
        df[f'AGE_RANGE_{min_age}_{max_age}'] = df['AGE'].apply(lambda x: 1 if min_age <= x <= max_age else 0)
    
    # Elimina la columna original de 'AGE'
    df.drop(columns=['AGE'], inplace=True)
    
    return df

def convert_payment_day(df):
    # Creamos las nuevas columnas
    for i in range(5, 30, 5):
        df[f'PAYMENT_DAY_{i}'] = df['PAYMENT_DAY'].apply(lambda x: 1 if i-5 < x <= i else 0)
    
    # Agregamos los valores más altos a la columna PAYMENT_DAY_25
    df['PAYMENT_DAY_25'] = df['PAYMENT_DAY'].apply(lambda x: 1 if x >= 25 else 0)

    # Eliminamos la columna original
    df.drop(columns=['PAYMENT_DAY'], inplace=True)
    
    return df