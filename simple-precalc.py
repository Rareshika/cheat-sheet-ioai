import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dir='' # bag aici dir
X = pd.read_csv(dir)

# Remove rows with missing target, separate target from predictors
target_feature='' # bag aici ce trebuie
X.dropna(axis=0, subset=[target_feature], inplace=True)
y = X.Price              
X.drop([target_feature], axis=1, inplace=True)


from sklearn.model_selection import train_test_split

X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

categorical_features = X_train.select_dtypes(include="object").columns
num_features = X_train.select_dtypes(exclude="object").columns


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print('Done!')


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print('transformat!')

import torch
from torch import nn

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train.to_numpy())
y_test = torch.from_numpy(y_test.to_numpy())
