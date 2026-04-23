from sklearn.metrics import classification_report, precision_recall_fscore_support

def evaluate(model, X_text_test, X_struct_test, y_test, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6]

    y_scores = model.predict([X_text_test, X_struct_test]).ravel()

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=[1],
            average=None,
            zero_division=0,
        )

        print(f"\n===== Threshold {threshold:.2f} =====")
        print(
            f"Fraude -> precision: {precision[0]:.4f} | "
            f"recall: {recall[0]:.4f} | f1: {f1[0]:.4f}"
        )
        print(classification_report(y_test, y_pred, zero_division=0))
