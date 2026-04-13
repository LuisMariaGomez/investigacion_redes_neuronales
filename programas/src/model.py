from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

def build_model(input_text_dim, input_struct_dim):
    
    # Input texto (embeddings)
    input_text = Input(shape=(input_text_dim,))
    
    # Input estructurado
    input_struct = Input(shape=(input_struct_dim,))
    
    # Capas para texto
    x1 = Dense(128, activation='relu')(input_text)
    x1 = Dense(64, activation='relu')(x1)
    
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