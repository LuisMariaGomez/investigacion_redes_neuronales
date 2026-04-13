"""
Entrada:
input_text_dim -> tamaño del embedding (ej: 384, tamaño del vector numérico que representa un texto)
input_struct_dim -> cantidad de variables numéricas (ej: 10, cantidad de columnas del dataset)
--------


"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

def build_model(input_text_dim, input_struct_dim):
    input_text = Input(shape=(input_text_dim,))     # Input texto (embeddings)
    input_struct = Input(shape=(input_struct_dim,)) # Input estructurado
    
    # Capas para texto
    x1 = Dense(128, activation='relu')(input_text) # transformamos el embedding en un vextor de 128 dimensiones
    x1 = Dense(64, activation='relu')(x1) # ahora este vector de 128 lo transformamos en uno de 64 dimensiones, que es mas chico y mas manejable para la parte estructurada (que es mas chica)
    
    # Capas para datos estructurados
    x2 = Dense(32, activation='relu')(input_struct)
    
    # Merge
    combined = Concatenate()([x1, x2])
    
    # Capas finales
    x = Dense(64, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_text, input_struct], outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model