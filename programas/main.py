from src.data_loader import load_data
from src.preprocessing import (
    formatear_texto,
    separar_texto_etiquetas,
)
from src.embeddings import HFEmbedder
from src.model import build_model
from imblearn.over_sampling import RandomOverSampler
from src.evaluate import evaluate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# asi veo que escribe y si lo hace bien o parece chino basico
cantidad_ejemplos_de_texto_formateado = 3
def print_formatted_examples(texts, labels, limit=3):
    print("\n===== Ejemplos de FormattedText =====")

    for index, (text, label) in enumerate(zip(texts.iloc[:limit], labels.iloc[:limit]), start=1):
        print(f"\nEjemplo {index} | isfraud={label}")
        print(text)
# 
def correr_experimento(
    tipo_de_experimento,
    X_train,
    y_train,
    X_test,
    y_test,
    usar_oversampler=False,
    class_weight=None,
):
    print(f"\n\n{'=' * 20} {tipo_de_experimento} {'=' * 20}")

    X_train_final = X_train
    y_train_final = y_train

    if usar_oversampler:
        oversampler = RandomOverSampler(random_state=42)
        X_train_final, y_train_final = oversampler.fit_resample(X_train, y_train)
        print("Balanceo aplicado: RandomOverSampler")
    else:
        print("Balanceo aplicado: ninguno")

    if class_weight is not None:
        print(f"class_weight aplicado: {class_weight}")
    else:
        print("class_weight aplicado: ninguno")

    print("Distribucion train original:")
    print(y_train.value_counts())
    print("Distribucion train usada para entrenar:")
    print(y_train_final.value_counts())

    model = build_model(X_train.shape[1])

    model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        class_weight=class_weight,
    )

    evaluate(model, X_test, y_test, thresholds=THRESHOLDS)

# Pipeline -----------------------------------------------------------------------------------

# Cargar datos
df = load_data("data/xlsx/Robo_Ruedas.xlsx")

# Construir un texto unico por fila usando todos los campos menos la etiqueta
df = formatear_texto(df, "isfraud")

# Separar features (texto enriquecido, etiqueta)
X_text, y = separar_texto_etiquetas(df, "FormattedText", "isfraud")

print_formatted_examples(X_text, y, limit=cantidad_ejemplos_de_texto_formateado)

print("\n===== Distribucion global de clases =====")
print(y.value_counts())

# Split train/test
X_text_train_raw, X_text_test_raw, y_train, y_test = train_test_split(
    X_text, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Embeddings (texto → vectores)
embedder = HFEmbedder()
X_text_train = embedder.encode(X_text_train_raw.tolist())
X_text_test = embedder.encode(X_text_test_raw.tolist())

classes = np.array(sorted(y_train.unique()))
balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)
class_weight = {cls: weight for cls, weight in zip(classes, balanced_weights)}

# Experimentos ---------------------------------------------------------------
correr_experimento(
    tipo_de_experimento="Experimento 1 - Sin balanceo",
    X_train=X_text_train,
    y_train=y_train,
    X_test=X_text_test,
    y_test=y_test,
)

correr_experimento(
    tipo_de_experimento="Experimento 2 - RandomOverSampler",
    X_train=X_text_train,
    y_train=y_train,
    X_test=X_text_test,
    y_test=y_test,
    usar_oversampler=True,
)

correr_experimento(
    tipo_de_experimento="Experimento 3 - class_weight",
    X_train=X_text_train,
    y_train=y_train,
    X_test=X_text_test,
    y_test=y_test,
    class_weight=class_weight,
)
