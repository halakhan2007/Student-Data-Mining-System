import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run():
    #l. Load Web Data
    data = pd.read_csv(
        "data/web_usage.csv"
    )

    features = [
        "Login_Count",
        "Pages_Visited",
        "Video_Minutes",
        "Quiz_Attempts",
        "Assignment_Downloads"
    ]

    X = data[features]

    #2. Standardize
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    #3. Create Behaviour Clusters
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    result = model.fit_predict(
        X_scaled
    )

    data["Behaviour_Cluster"] = result

    #4. Summary
    summary = data.groupby(
        "Behaviour_Cluster"
    )[features].mean().round(2)

    return {
        "data": data,
        "summary": summary
    }