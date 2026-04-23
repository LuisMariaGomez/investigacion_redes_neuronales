from src.data_loader import load_data
from src.preprocessing import (
    build_structured_preprocessor,
    clean_text,
    split_features,
    transform_structured_features,
)
from src.embeddings import HFEmbedder
from src.model import build_model
import numpy as np
from imblearn.over_sampling import RandomOverSampler

# para ver como esta evaluando (despues del primer intento)

from src.evaluate import evaluate
from sklearn.model_selection import train_test_split

# Cargar datos
df = load_data("data/xlsx/Robo_Ruedas.xlsx")

# Limpiar texto, aca mandarle el nombre de la columna que tenga nan
df = clean_text(df, "TextoDenuncia")


# Separar features (texto, datos estructurados, etiqueta)
X_text, X_struct, y = split_features(df, "TextoDenuncia", "isfraud")

# para ver que tan desbalanceado esta
# print("Distribución de clases:")
# print(y.value_counts())

# 6. Split antes de transformar para evitar fuga de información
X_text_train_raw, X_text_test_raw, X_struct_train_raw, X_struct_test_raw, y_train, y_test = train_test_split(
    X_text, X_struct, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Embeddings (texto → vectores)
embedder = HFEmbedder()
X_text_train = embedder.encode(X_text_train_raw.tolist())
X_text_test = embedder.encode(X_text_test_raw.tolist())

# Procesar datos estructurados: numéricos + categóricos con One-Hot Encoding
struct_preprocessor, categorical_columns, numeric_columns = build_structured_preprocessor(
    X_struct_train_raw
)
X_struct_train, X_struct_test = transform_structured_features(
    struct_preprocessor,
    X_struct_train_raw,
    X_struct_test_raw,
    categorical_columns,
    numeric_columns,
)

print("Columnas numéricas:", numeric_columns)
print("Columnas categóricas codificadas con One-Hot:", categorical_columns)
print("Dimensión estructurada tras encoding:", X_struct_train.shape[1])

# Aplicar oversampling solo sobre entrenamiento
oversampler = RandomOverSampler(random_state=42)
X_train_combined = np.hstack([X_text_train, X_struct_train])
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_combined, y_train)

text_dim = X_text_train.shape[1]
X_text_train_resampled = X_train_resampled[:, :text_dim]
X_struct_train_resampled = X_train_resampled[:, text_dim:]

# 7. Crear modelo
model = build_model(X_text_train.shape[1], X_struct_train.shape[1])


# Entrenar (con balanceo) despues emterlo en train.py
history = model.fit(
    [X_text_train_resampled, X_struct_train_resampled],
    y_train_resampled,
    validation_data=([X_text_test, X_struct_test], y_test),
    epochs=10,
    batch_size=32
)

# 10. Evaluar 🔥
evaluate(model, X_text_test, X_struct_test, y_test)
