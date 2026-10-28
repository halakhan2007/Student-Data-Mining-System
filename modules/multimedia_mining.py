import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run():
    #l. Load Web Data
    data = pd.read_csv(
        "data/multimedia_data.csv"
    )

    features = [
        "Brightness",
        "Size",
        "Duration"
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

    data["Feature_Cluster"] = result

    #4. Summary
    summary = data.groupby(
        "Feature_Cluster"
    )[features].mean().round(2)

    return {
        "data": data,
        "summary": summary
    }