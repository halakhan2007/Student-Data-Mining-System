import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
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
        test_size=0.30,
        random_state=42
    )

    # 3. Create Decision Tree
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )
    # 4. Train
    model.fit(
        X_train,
        y_train
    )

    # 5. Prediction
    result = model.predict(
        X_test
    )

    # 6. Accuracy
    accuracy = accuracy_score(
        y_test,
        result
    ) * 100

    # 7. Tree Information
    tree_depth = model.get_depth()
    leaves = model.get_n_leaves()

    # 8. Root Attribute
    root_index = model.tree_.feature[0]

    if root_index >= 0:
        root_attribute = features[root_index]
    else:
        root_attribute = "N/A"

    # 9. Decision Rules
    rules = export_text(
        model,
        feature_names=features
    )
    # 10. Prediction Table
    output = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": result
    })

    return {
        "data": output,
        "accuracy": accuracy,
        "criterion": "Gini",
        "tree_depth": tree_depth,
        "leaves": leaves,
        "root_attribute": root_attribute,
        "rules": rules
    }