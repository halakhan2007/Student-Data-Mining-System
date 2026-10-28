import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run():
    #l. Load Web Data
    data = pd.read_csv(
        "data/spatial_data.csv"
    )

    features = [
        "Latitude",
        "Longitude",
        "Students",
        "Attendance",
        "Average_Marks"
    ]

    X = data[features]

    #2. Standardize
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    #3. Create Behaviour Clusters
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    result = model.fit_predict(
        X_scaled
    )

    data["Spatial_Cluster"] = result

    #4. Summary
    summary = data.groupby(
        "Spatial_Cluster"
    )[
        [
            "Students",
            "Attendance",
            "Average_Marks"
        ]
    ].mean().round(2)

    return {
        "data": data,
        "summary": summary
    }