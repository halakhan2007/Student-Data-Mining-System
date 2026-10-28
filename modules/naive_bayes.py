import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
 
def run(): 
 
    # 1. Load Data 
    data = pd.read_csv( 
        "data/student_data.csv" 
    ) 
 
    features = [ 
        "Study_Hours", 
        "Attendance", 
        "Assignment_Score", 
        "Internal_Marks", 
        "Previous_Score" 
    ] 
 
    X = data[features] 
    y = data["Result"] 
 
    # 2. Split Data 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, 
        y, 
        test_size=0.20, 
        random_state=42 
    ) 
 
    # 3. Create Model 
    model = GaussianNB() 
 
    # 4. Train
    model.fit( 
        X_train, 
        y_train 
    ) 
 
    # 5. Prediction 
    result = model.predict( 
        X_test 
    ) 
 
    # 6. Probability 
    probability = model.predict_proba( 
        X_test 
    ) 
 
    # 7. Accuracy 
    accuracy = accuracy_score( 
        y_test, 
        result 
    ) * 100 
 
    # 8. Output 
    output = pd.DataFrame({ 
        "Actual": y_test.values, 
        "Predicted": result 
    }) 
 
    for i, class_name in enumerate( 
        model.classes_ 
    ): 
 
        output[ 
            class_name + "_Probability" 
        ] = ( 
            probability[:, i] * 100 
        ).round(2) 
 
    return { 
        "data": output, 
        "accuracy": accuracy, 
        "classes": list(model.classes_) 
    } 