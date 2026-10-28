import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

    x = data[features]

    X_scaled = StandardScaler().fit_transform(x)
    #KMeans
    k = 3
    model = KMeans(
        n_clusters = k,
        random_state = 42,
        n_init = 10
    )

    data["Cluster"] = model.fit_predict(X_scaled)

    # Cluster Summary
    summary = data.groupby("Cluster")[features].mean().round(2)

    return {
        "data": data,
        "summary": summary,
        "clusters": k
    }
