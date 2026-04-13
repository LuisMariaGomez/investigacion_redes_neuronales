from sklearn.metrics import classification_report

def evaluate(model, X_text_test, X_struct_test, y_test):
    y_pred = model.predict([X_text_test, X_struct_test])
    y_pred = (y_pred > 0.4).astype(int)

    print(classification_report(y_test, y_pred))