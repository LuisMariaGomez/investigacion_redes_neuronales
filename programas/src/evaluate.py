from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

def evaluate(model, X_text_test, y_test, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    y_scores = model.predict(X_text_test).ravel()

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
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        print(
            "Matriz de confusion -> "
            f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}"
        )
        print(classification_report(y_test, y_pred, zero_division=0))
