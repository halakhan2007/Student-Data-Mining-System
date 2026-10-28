import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def run():

    # 1. Load Data
    data = pd.read_csv(
        "data/student_data.csv"
    )

    # 2. Select Features
    features = [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Score"
    ]

    X = data[features]
    y = data["Result"]

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    # 4. Create ID3 Model
    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    )

    # 5. Train Model
    model.fit(
        X_train,
        y_train
    )

    # 6. Prediction
    result = model.predict(
        X_test
    )

    # 7. Accuracy
    accuracy = accuracy_score(
        y_test,
        result
    ) * 100
    # 8. Feature Importance
    importance = pd.DataFrame({
        "Attribute": features,
        "Information_Gain_Importance":
            model.feature_importances_
    })
    importance[
        "Information_Gain_Importance"
    ] = importance[
        "Information_Gain_Importance"
    ].round(5)
    
    importance = importance.sort_values(
        "Information_Gain_Importance",
        ascending=False
    )
    # 9. Root Attribute
    root_index = model.tree_.feature[0]

    if root_index >= 0:
        root_attribute = features[root_index]
    else:
        root_attribute = "N/A"

    # 10. Prediction Table
    output = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": result
    })

    return {
        "data": output,
        "accuracy": accuracy,
        "importance": importance,
        "root_attribute": root_attribute,
        "criterion": "Entropy"
    }