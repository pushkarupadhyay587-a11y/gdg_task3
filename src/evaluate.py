
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)

from .config import LABELS, REPORT_DIR


def evaluate(model, X, y, name: str = "test", save: bool = True) -> dict:
    pred = model.predict(X)
    names = [LABELS[i] for i in sorted(LABELS)]
    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
    }
    if save:
        REPORT_DIR.mkdir(exist_ok=True)
        text = classification_report(y, pred, target_names=names, digits=4)
        (REPORT_DIR / f"{name}_classification_report.txt").write_text(text)
        (REPORT_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        cm = confusion_matrix(y, pred, labels=sorted(LABELS))
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion matrix ({name})")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"{name}_confusion_matrix.png", dpi=120)
        plt.close(fig)
        print(text)
    return metrics
