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
    X_text = df[text_column]    # texto sin procesar de la denuncia
    y = df[target_column]       # columna de etiquetas
    X_struct = df.drop(columns=[text_column, target_column]) # volamos texto y etiqueta, queda solo lo estructurado
    
    return X_text, X_struct, y