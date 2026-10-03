import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, REPORT_DIR, VAL_SIZE
    from src.data import load_test, load_train
    from src.evaluate import evaluate
    from src.models import candidate_pipelines
else:
    from .config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, REPORT_DIR, VAL_SIZE
    from .data import load_test, load_train
    from .evaluate import evaluate
    from .models import candidate_pipelines


def main():
    train, test = load_train(), load_test()
    X_tr, X_val, y_tr, y_val = train_test_split(
        train["text"], train["label"], test_size=VAL_SIZE,
        stratify=train["label"], random_state=RANDOM_STATE)

    results = {}
    for name, pipe in candidate_pipelines().items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        results[name] = evaluate(pipe, X_val, y_val, save=False)
        results[name]["fit_seconds"] = round(time.time() - t0, 1)
        print(f"[validation] {name}: {results[name]}")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\nBest model on validation: {best}")

    final = candidate_pipelines()[best]
    final.fit(train["text"], train["label"])      
    print("\n=== Held-out TEST set ===")
    test_metrics = evaluate(final, test["text"], test["label"], name="test")
    print(test_metrics)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "model_comparison.json").write_text(json.dumps(
        {"validation": results, "selected": best, "test": test_metrics}, indent=2))
    print(f"Saved model -> {MODEL_PATH} ({MODEL_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()