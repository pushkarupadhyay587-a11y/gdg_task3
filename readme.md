# News Classifier API

A 4-class news topic classification system built with scikit-learn and served through a [FastAPI](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/app/app.py) REST API (`News Classifier API`, v1.0.0). The pipeline preprocesses article headlines and descriptions, extracts TF-IDF n-gram features, evaluates multiple linear and probabilistic classifiers, and serves predictions from the best-performing model bundle.

## Categories

Defined in [`src/config.py`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/src/config.py):

| Class Index | Category |
| :---: | :--- |
| `1` | World |
| `2` | Sports |
| `3` | Business |
| `4` | Science/Technology |

## Project Structure

```text
gdg_task3/
├── app/
│   ├── __init__.py
│   ├── app.py            # FastAPI application entry point & lifespan loader
│   ├── config.py         # Artifact paths & API title/version settings
│   ├── prediction.py     # NewsPredictor class for loading model & predicting
│   ├── routes.py         # API endpoints (/health, /model, /predict)
│   └── schemas.py        # Pydantic request/response schemas
├── artifacts/
│   ├── models/           # Individual trained model pickles (e.g., LinearSVC().pkl)
│   ├── best_model.joblib # Bundle with best model, vectorizer, name, & val F1
│   ├── preprocessor.pkl
│   └── vectoriser.pkl
├── data/
│   └── raw/              # Expected location for train.csv and test.csv
├── notebook/
│   └── data_cleaning.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py         # Dataset paths, random state, val size, class labels
│   ├── data.py           # Dataset loading helpers
│   ├── evaluate.py       # Test-set evaluation helper
│   ├── features.py       # TF-IDF vectorizer configuration
│   ├── preprocess.py     # Text cleaning & preprocessing pipeline
│   ├── report.md         # Model training & evaluation report
│   └── train.py          # Training, validation, evaluation, and artifact saving
└── requirements.txt      # Python dependencies
```

## Machine Learning Pipeline

### Dataset & Preprocessing
- **Dataset:** 120,000 training articles (`data/raw/train.csv`, balanced across 4 classes), 12,000 stratified validation articles (10% split with `random_state=42` in [`src/train.py`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/src/train.py)), and 7,600 test articles (`data/raw/test.csv`).
- **Text Cleaning ([`src/preprocess.py`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/src/preprocess.py)):** `preprocessor(df)` lowercases `Title`, removes trailing parentheticals, cleans backslashes and leading prefixes in `Description`, strips HTML tags, URLs, punctuation, and `ENGLISH_STOP_WORDS`, and combines the cleaned fields into a `clean` text column.
- **Feature Extraction ([`src/features.py`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/src/features.py)):** `TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=100000, sublinear_tf=True)` fitted on training data only.

### Model Performance

Results from [`src/report.md`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/src/report.md):

| Model | Validation Accuracy | Test Accuracy | World F1 | Sports F1 | Business F1 | Sci/Tech F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `LinearSVC()` | 0.93 | 0.92 | 0.93 | 0.97 | 0.90 | 0.90 |
| `LogisticRegression(C=10, max_iter=1000, random_state=42)` | 0.93 | 0.92 | 0.92 | 0.97 | 0.89 | 0.90 |
| `ComplementNB()` | 0.92 | 0.91 | 0.91 | 0.97 | 0.87 | 0.89 |

## Getting Started

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/pushkarupadhyay587-a11y/gdg_task3.git
cd gdg_task3
pip install -r requirements.txt
```

Core dependencies in [`requirements.txt`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/requirements.txt): `fastapi[standard]`, `pydantic`, `pandas`, `scikit-learn`, `joblib`, and `matplotlib`.

### 2. Train the Models (Optional)

Pre-trained artifacts are included under `artifacts/`. To retrain from `data/raw/train.csv` and `data/raw/test.csv` and regenerate `artifacts/best_model.joblib`:

```bash
python -m src.train
```

### 3. Run the API Server

Start the FastAPI development server:

```bash
fastapi dev app/app.py
```

Or run with Uvicorn:

```bash
uvicorn app.app:app --reload
```

On startup, [`app/prediction.py`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/app/prediction.py) loads `artifacts/best_model.joblib` (with a legacy fallback to `artifacts/models/LinearSVC().pkl` and `artifacts/vectoriser.pkl`). Interactive API docs are available at `/docs` once the server is running.

## API Endpoints

Defined in [`app/routes.py`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/app/routes.py) and [`app/schemas.py`](https://github.com/pushkarupadhyay587-a11y/gdg_task3/blob/main/app/schemas.py):

### `GET /health`
Health check endpoint.

```json
{
  "status": "ok"
}
```

### `GET /model`
Returns the active model name and its validation macro F1 score (`ModelInfo`).

```json
{
  "model": "LinearSVC",
  "validation_macro_f1": 0.93
}
```

### `POST /predict`
Classifies a news article from its headline and description (`NewsArticle` -> `PredictionResult`). Both fields require a minimum length of 1 character.

**Request Body:**
```json
{
  "title": "Tech Giant Unveils Next-Gen AI Chip",
  "description": "The new processor promises major efficiency gains for data center workloads."
}
```

**Response Body:**
```json
{
  "class_index": 4,
  "category": "Science/Technology"
}
```
