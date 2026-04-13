from sklearn.model_selection import train_test_split

def train_model(model, X_text, X_struct, y):

    X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
        X_text, X_struct, y, test_size=0.2, random_state=42
    )

    history = model.fit(
        [X_text_train, X_struct_train],
        y_train,
        validation_data=([X_text_test, X_struct_test], y_test),
        epochs=10,
        batch_size=32
    )

    return model, history