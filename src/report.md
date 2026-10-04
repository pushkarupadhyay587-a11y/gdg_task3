

Model training report · MD
# Model Training Report
 
News topic classification (4 classes) using TF-IDF features.
 
## Setup
 
- **Data:** 120,000 training articles, 7,600 test articles, classes perfectly balanced
- **Validation:** 12,000 articles (10% of train, stratified)
- **Features:** TF-IDF, unigrams + bigrams, 100,000 features, fitted on training data only
- **Models:** LinearSVC, Logistic Regression, Complement Naive Bayes
## Results
 
| Model | Validation accuracy | Test accuracy |
|---|---|---|
| LinearSVC | 0.93 | **0.92** |
| Logistic Regression | 0.93 | **0.92** |
| Complement Naive Bayes | 0.92 | 0.91 |
 
Validation and test scores are within one point of each other, so the models are not overfitting.
 
## Test F1 by class
 
| Class | LinearSVC | Logistic Regression | Complement NB |
|---|---|---|---|
| 1 | 0.93 | 0.92 | 0.91 |
| 2 | 0.97 | 0.97 | 0.97 |
| 3 | 0.90 | 0.89 | 0.87 |
| 4 | 0.90 | 0.90 | 0.89 |
 
- Class 2 is the easiest to predict.
- Class 3 is the hardest.
## Conclusion
 
LinearSVC is the best choice. It ties for the top accuracy (0.92) and trains fast. Logistic Regression is a good alternative if you need probabilities.
 
> **Note:** an earlier test run gave about 0.25 accuracy because the vectorizer was refitted on the test data. It must be fitted on training data only and then used with `transform` on test data.
 
