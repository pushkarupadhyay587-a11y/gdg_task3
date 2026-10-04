from .data import load_test 
from .preprocess import preprocessor  
from sklearn.metrics import classification_report

def evaluater(model, vec):
    dft= load_test()
    preprocessor(dft)
    dft = dft.drop(["Title","Description"],axis=1)
    
    x_test = dft["clean"]
    y_test = dft["Class Index"]
    
    x_test_vec = vec.transform(x_test)
    
    y_prd = model.predict(x_test_vec)
    print(f"{model} : \n",classification_report(y_test, y_prd))
    
    