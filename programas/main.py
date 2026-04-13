from src.data_loader import load_data
from src.preprocessing import clean_text, split_features
from src.embeddings import HFEmbedder
from src.model import build_model
from src.train import train_model
import numpy as np


# para ver como esta evaluando (despues del primer intento)

from src.evaluate import evaluate
from sklearn.model_selection import train_test_split




# Cargar datos
df = load_data("data/raw/Robo_Ruedas.xlsx")

# Limpiar texto, aca mandarle el nombre de la columna que tenga nan
df = clean_text(df, "TextoDenuncia")


# Separar features (texto, datos estructurados, etiqueta)
X_text, X_struct, y = split_features(df, "TextoDenuncia", "isfraud")

# para ver que tan desbalanceado esta
# print("Distribución de clases:")
# print(y.value_counts())

# Embeddings (texto → vectores)
embedder = HFEmbedder()
X_text_emb = embedder.encode(X_text.tolist()) # aca devolveria algo de estilo [54411, 384], el 384 es el tamaño del vector que representa un texto (el que vamos a usar), 54411 es la cantidad de textos

# Procesar datos estructurados (solo numéricos)
X_struct = X_struct.select_dtypes(include=[np.number]).fillna(0).values # devuleve algo como [54411, 10], 10 es la cantidad de variables numéricas

# Split (IMPORTANTE: después de todo el procesamiento)
X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_text_emb, X_struct, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Crear modelo
model = build_model(X_text_emb.shape[1], X_struct.shape[1])


# Entrenar (con balanceo) despues emterlo en train.py
history = model.fit(
    [X_text_train, X_struct_train],
    y_train,
    validation_data=([X_text_test, X_struct_test], y_test),
    epochs=10,
    batch_size=32,
    class_weight = {0: 1, 1: 20}
)

# 10. Evaluar 🔥
evaluate(model, X_text_test, X_struct_test, y_test) 