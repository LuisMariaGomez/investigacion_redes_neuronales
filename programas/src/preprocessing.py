"""
Aca la onda es separar:
    texto
    datos estructurados
    etiqueta
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def clean_text(df, text_column):
    df[text_column] = df[text_column].fillna("").str.lower()
    return df

def split_features(df, text_column, target_column):
    X_text = df[text_column]    # texto sin procesar de la denuncia
    y = df[target_column]       # columna de etiquetas
    X_struct = df.drop(columns=[text_column, target_column]) # volamos texto y etiqueta, queda solo lo estructurado
    
    return X_text, X_struct, y


def build_structured_preprocessor(X_struct_train):
    categorical_columns = X_struct_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_columns = X_struct_train.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )

    return preprocessor, categorical_columns, numeric_columns


def transform_structured_features(
    preprocessor,
    X_struct_train,
    X_struct_test,
    categorical_columns,
    numeric_columns,
):
    X_struct_train = X_struct_train.copy()
    X_struct_test = X_struct_test.copy()

    X_struct_train[categorical_columns] = X_struct_train[categorical_columns].fillna("missing")
    X_struct_test[categorical_columns] = X_struct_test[categorical_columns].fillna("missing")
    X_struct_train[numeric_columns] = X_struct_train[numeric_columns].fillna(0)
    X_struct_test[numeric_columns] = X_struct_test[numeric_columns].fillna(0)

    X_struct_train_encoded = preprocessor.fit_transform(X_struct_train)
    X_struct_test_encoded = preprocessor.transform(X_struct_test)

    return X_struct_train_encoded, X_struct_test_encoded
