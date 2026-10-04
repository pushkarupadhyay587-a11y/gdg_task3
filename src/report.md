## training report 

=== Evaluation Report ===
              precision    recall  f1-score   support

           1       0.94      0.92      0.93      3000
           2       0.96      0.98      0.97      3000
           3       0.90      0.90      0.90      3000
           4       0.91      0.91      0.91      3000

    accuracy                           0.93     12000
   macro avg       0.93      0.93      0.93     12000
weighted avg       0.93      0.93      0.93     12000

model :  LogisticRegression(C=10, max_iter=1000, random_state=42) 


=== Evaluation Report ===
              precision    recall  f1-score   support

           1       0.93      0.92      0.93      3000
           2       0.97      0.98      0.97      3000
           3       0.90      0.90      0.90      3000
           4       0.91      0.91      0.91      3000

    accuracy                           0.93     12000
   macro avg       0.93      0.93      0.93     12000
weighted avg       0.93      0.93      0.93     12000

model :  ComplementNB() 


=== Evaluation Report ===
              precision    recall  f1-score   support

           1       0.93      0.91      0.92      3000
           2       0.94      0.99      0.97      3000
           3       0.90      0.88      0.89      3000
           4       0.90      0.90      0.90      3000

    accuracy                           0.92     12000
   macro avg       0.92      0.92      0.92     12000
weighted avg       0.92      0.92      0.92     12000

====================================================================
## testing report 

LinearSVC() : 
               precision    recall  f1-score   support

           1       0.94      0.91      0.93      1900
           2       0.96      0.99      0.97      1900
           3       0.90      0.89      0.90      1900
           4       0.89      0.91      0.90      1900

    accuracy                           0.92      7600
   macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600

LogisticRegression(C=10, max_iter=1000, random_state=42) : 
               precision    recall  f1-score   support

           1       0.93      0.91      0.92      1900
           2       0.96      0.98      0.97      1900
           3       0.89      0.89      0.89      1900
           4       0.89      0.90      0.90      1900

    accuracy                           0.92      7600
   macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600

ComplementNB() : 
               precision    recall  f1-score   support

           1       0.93      0.90      0.91      1900
           2       0.94      0.99      0.97      1900
           3       0.89      0.86      0.87      1900
           4       0.89      0.89      0.89      1900

    accuracy                           0.91      7600
   macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600