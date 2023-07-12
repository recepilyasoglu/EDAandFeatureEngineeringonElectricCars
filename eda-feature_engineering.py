import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use("Qt5Agg")

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

data = pd.read_csv("Electric_Vehicle_Population_Data.csv")
df = data.copy()
df.head()

# EDA
def get_stats(dataframe):
    return print("############### First 5 Line ############### \n", dataframe.head(), "\n", \
                 "############### Number of Values Owned ############### \n", dataframe.value_counts(), "\n", \
                 "############### Total Number of Observations ############### \n", dataframe.shape, "\n", \
                 "############### Variables Types ############### \n", dataframe.dtypes, "\n", \
                 "############### Total Number of Null Values ############### \n", dataframe.isnull().sum(), "\n", \
                 "############### Descriptive Statistics ############### \n", dataframe.describe().T
                 )

get_stats(df)


def grab_col_names(dataframe, cat_th=2, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df[cat_cols]
df[num_cols].head()
df[cat_but_car].head()


# analysis of categorical variables
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###############################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)


# analysis of numerical variables
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, True)


# outliers
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)


def check_outlier(dataframe, col_name):  # q1 ve q3 'ü de biçimlendirmek istersek check_outlier'a argüman olarak girmemiz gerekir
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False   # var mı yok mu sorusuna bool dönmesi lazım (True veya False)


for col in num_cols:
    print(col, check_outlier(df, col))


# analysis missing values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


# Feature Engineering

df.dtypes
df["Model"].head()

df.rename(columns={'County': 'Country'}, inplace=True)
df["Country"]


# applying missing values
df.isnull().sum()

df[["Country", "City", "Postal Code", "Model", "Legislative District",
    "Vehicle Location", "Electric Utility", "2020 Census Tract"]].head()

# Since there are few null values in the categorical variables Country and City,
# I decided to drop them directly from the data set.
def del_null(dataframe, variable):
    dataframe.dropna(subset=variable, inplace=True)

del_null(df, ["Country", "City"])

df.dtypes

# instead of filling in missing values in categorical variables i replaced it with 0
values = ["Model", "Vehicle Location", "Electric Utility"]

df[values] = df[values].fillna(0)


# I chose to fill the numeric variable with its mean
df["Legislative District"].head()
df["Legislative District"] = df["Legislative District"].fillna(df["Legislative District"].mean())

df.isnull().sum()

df.columns


# Creating new variables

# Age
current_year = 2023
df['Car Age'] = current_year - df['Model Year']

# Fuel Efficiency
bins = [0, 100, 200, 300, float('inf')]
labels = ['Short Range', 'Medium Range', 'Long Range', 'Very Long Range']

df['Fuel Efficiency'] = pd.cut(df['Electric Range'], bins=bins, labels=labels, right=False)

# Electric Utility Feature
df['Electric Utility Feature'] = df['Electric Utility'].apply(lambda x: 'Available' if x != 'None' else "Not Available")

df.head()

