import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from .data import load_train,load_test
from .preprocess import preprocessor
from .features import build_vectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,ComplementNB
from .config import RANDOM_STATE
from .evaluate import evaluater
from .config import MODEL_PATH

def main():

    df = load_train()
    vec = build_vectorizer()
    
    preprocessor(df)
    df = df.drop(["Title","Description"],axis=1)
    
    x = df["clean"]
    y = df["Class Index"]
    
    x_vec = vec.fit_transform(x)
    
    x_train, x_val, y_train, y_val = train_test_split(
    x_vec, y, test_size=0.1, stratify=y, random_state=42)
    
    
    svc_model  = LinearSVC()
    lr_model = LogisticRegression(C=10, max_iter=1000, random_state=RANDOM_STATE)
    nb_model = ComplementNB()


    print("Training naive bayes model...")
    nb_model.fit(x_train,y_train)
    print("Training linear regression model...")
    lr_model.fit(x_train,y_train)
    print("Training support_vector model...")
    svc_model.fit(x_train,y_train)
     
     
    for model in [svc_model,lr_model,nb_model]:
        preds = model.predict(x_val)
        print("model : ",model,"\n")
        print("\n=== Evaluation Report ===")
        print(classification_report(y_val, preds))
        
        
    print("====================================================================")
    for model in [svc_model,lr_model,nb_model]:
        evaluater(model, vec)
        
    for model in [svc_model,lr_model,nb_model]:
        joblib.dump(model, MODEL_PATH/"models"/f"{model}.pkl")
    
    joblib.dump(vec,MODEL_PATH/"vectoriser.pkl")
    joblib.dump(preprocessor,MODEL_PATH/"preprocessor.pkl")


if __name__ == "__main__":
    main()