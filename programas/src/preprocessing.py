"""
Aca la onda es separar:
    texto
    datos estructurados
    etiqueta
"""

def clean_text(df, text_column):
    df[text_column] = df[text_column].fillna("").str.lower()
    return df

def split_features(df, text_column, target_column):
    X_text = df[text_column]
    y = df[target_column]
    
    X_struct = df.drop(columns=[text_column, target_column])
    
    return X_text, X_struct, y