import streamlit as st

from modules import id3
from modules import j48
from modules import naive_bayes
from modules import decision_tree
from modules import knn
from modules import kmeans
from modules import apriori
from modules import text_mining
from modules import web_mining
from modules import multimedia_mining
from modules import spatial_mining


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Data Mining System",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# PROJECT TITLE
# =========================================================

st.title("🎓 Student Intelligence and Learning Analytics System")

st.write(
    """
    This project demonstrates different Data Mining techniques
    using student academic, learning, feedback, web, multimedia
    and spatial data.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Data Mining Techniques")

technique = st.sidebar.selectbox(
    "Select Technique",
    [
        "Home",
        "ID3",
        "J48",
        "Naive Bayes",
        "Decision Tree",
        "K-NN",
        "K-Means",
        "Apriori",
        "Text Mining",
        "Web Mining",
        "Multimedia Mining",
        "Spatial Mining"
    ]
)


# =========================================================
# HOME
# =========================================================

if technique == "Home":

    st.header("🏠 Project Dashboard")

    st.write(
        """
        The system applies different Data Mining techniques
        to solve different types of problems in education.
        """
    )

    st.subheader("📚 Techniques Covered")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            🌳 **ID3**

            Classification

            Main Concept:
            Information Gain
            """
        )

        st.info(
            """
            🌳 **J48**

            Classification

            Main Concept:
            C4.5 Decision Tree
            """
        )

        st.info(
            """
            🎲 **Naive Bayes**

            Classification

            Main Concept:
            Probability
            """
        )

        st.info(
            """
            🌳 **Decision Tree**

            Classification

            Main Concept:
            Gini / Decision Rules
            """
        )

    with col2:

        st.info(
            """
            👥 **K-NN**

            Classification

            Main Concept:
            Distance / Neighbours
            """
        )

        st.info(
            """
            🔵 **K-Means**

            Clustering

            Main Concept:
            Clusters / Centroids
            """
        )

        st.info(
            """
            🛒 **Apriori**

            Association Mining

            Main Concept:
            Support / Confidence / Lift
            """
        )

        st.info(
            """
            📝 **Text Mining**

            Text Analysis

            Main Concept:
            Frequency / TF-IDF
            """
        )

    with col3:

        st.info(
            """
            🌐 **Web Mining**

            Web Analysis

            Main Concept:
            Web Behaviour
            """
        )

        st.info(
            """
            🎥 **Multimedia Mining**

            Multimedia Analysis

            Main Concept:
            Feature Groups
            """
        )

        st.info(
            """
            📍 **Spatial Mining**

            Spatial Analysis

            Main Concept:
            Location / Area Patterns
            """
        )

    st.subheader("🎯 Project Objectives")

    st.write(
        """
        • Predict whether a student is likely to Pass or Fail.

        • Classify students using different classification algorithms.

        • Find groups of similar students.

        • Discover relationships between learning activities.

        • Analyze student feedback.

        • Analyze student behaviour on an online learning portal.

        • Group educational multimedia resources.

        • Identify patterns in different geographical areas.
        """
    )

    st.subheader("🔄 Common Data Mining Flow")

    st.code(
        """
Load Data
    ↓
Select Features
    ↓
Create Model
    ↓
Apply Algorithm
    ↓
Generate Result
    ↓
Interpret Result
        """,
        language="text"
    )


# =========================================================
# ID3
# =========================================================

# =========================================================
# ID3
# =========================================================
elif technique == "ID3":

    st.header("🌳 ID3 Classification")

    st.write(
        """
        ID3 is a decision tree classification algorithm.
        It uses Entropy and Information Gain to select
        the best attribute for splitting the data.
        """
    )

    if st.button("▶ Run ID3", key="run_id3"):

        result = id3.run()

        st.success("✅ ID3 executed successfully!")

        # -------------------------------------------------
        # MODEL INFORMATION
        # -------------------------------------------------

        st.subheader("📊 Model Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{result.get('accuracy', 0):.2f}%"
            )

        with col2:
            st.metric(
                "Algorithm",
                "ID3"
            )

        with col3:
            st.metric(
                "Criterion",
                result.get("criterion", "Entropy")
            )

        with col4:
            st.metric(
                "Root Attribute",
                result.get("root_attribute", "N/A")
            )

        # -------------------------------------------------
        # PREDICTION RESULT
        # -------------------------------------------------

        st.subheader("📋 Prediction Result")

        prediction_data = result.get("data")

        if prediction_data is not None:
            st.dataframe(
                prediction_data,
                use_container_width=True
            )

        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        st.subheader("📈 Attribute Importance")

        importance_data = result.get("importance")

        if importance_data is not None:
            st.dataframe(
                importance_data,
                use_container_width=True
            )

        # -------------------------------------------------
        # IMPORTANT ATTRIBUTE
        # -------------------------------------------------

        st.subheader("🌱 Root Attribute")

        root_attribute = result.get(
            "root_attribute",
            "N/A"
        )

        st.info(
            f"""
            The root attribute selected by the decision tree is:

            **{root_attribute}**
            """
        )

        # -------------------------------------------------
        # CONCEPT EXPLANATION
        # -------------------------------------------------

        st.subheader("💡 What did we learn?")

        st.info(
            """
            **ID3 follows these basic steps:**

            1. Calculate Entropy
            2. Calculate Information Gain
            3. Select the best attribute
            4. Create the Decision Tree
            5. Predict Pass / Fail

            The attribute with the best split becomes an
            important decision point in the tree.
            """
        )



# =========================================================
# J48
# =========================================================

elif technique == "J48":

    st.header("🌳 J48 Classification")

    st.write(
        """
        J48 is Weka's implementation of the C4.5 decision tree.
        In this Python project, an entropy-based decision tree
        is used as a C4.5-style implementation.
        """
    )

    if st.button("▶ Run J48"):

        result = j48.run()

        st.success("J48 executed successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Accuracy",
                f"{result.get('accuracy', 0):.2f}%"
            )

        with col2:

            st.metric(
                "Tree Depth",
                result.get("tree_depth", "N/A")
            )

        with col3:

            st.metric(
                "Number of Leaves",
                result.get("leaves", "N/A")
            )

        st.subheader("📋 Prediction Result")

        st.dataframe(
            result.get("data"),
            use_container_width=True
        )

        st.subheader("💡 What did we learn?")

        st.info(
            """
            J48 is based on the C4.5 decision tree approach.

            Tree Depth tells us how deep the tree is.

            Number of Leaves tells us how many final decision
            nodes are present.
            """
        )


# =========================================================
# NAIVE BAYES
# =========================================================

elif technique == "Naive Bayes":

    st.header("🎲 Naive Bayes Classification")

    st.write(
        """
        Naive Bayes uses probability to predict
        the class of a student.
        """
    )

    if st.button("▶ Run Naive Bayes"):

        result = naive_bayes.run()

        st.success("Naive Bayes executed successfully!")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Accuracy",
                f"{result.get('accuracy', 0):.2f}%"
            )

        with col2:

            st.metric(
                "Main Concept",
                "Probability"
            )

        st.subheader("📋 Prediction with Probability")

        st.dataframe(
            result.get("data"),
            use_container_width=True
        )

        st.subheader("💡 What did we learn?")

        st.info(
            """
            Naive Bayes calculates the probability of each class.

            The class having the highest probability
            is selected as the prediction.
            """
        )


# =========================================================
# DECISION TREE
# =========================================================

elif technique == "Decision Tree":

    st.header("🌳 Decision Tree Classification")

    st.write(
        """
        Decision Tree creates decision rules and uses
        them to classify students.
        """
    )

    if st.button("▶ Run Decision Tree"):

        result = decision_tree.run()

        st.success("Decision Tree executed successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Accuracy",
                f"{result.get('accuracy', 0):.2f}%"
            )

        with col2:

            st.metric(
                "Criterion",
                result.get("criterion", "Gini")
            )

        with col3:

            st.metric(
                "Tree Depth",
                result.get("tree_depth", "N/A")
            )

        st.subheader("📋 Prediction Result")

        st.dataframe(
            result.get("data"),
            use_container_width=True
        )

        st.subheader("💡 What did we learn?")

        st.info(
            """
            Decision Tree divides data using decision rules.

            In this practical, Gini is used as the splitting criterion.
            """
        )


# =========================================================
# K-NN
# =========================================================

elif technique == "K-NN":

    st.header("K-Nearest Neighbours (K-NN)")

    result = knn.run()

    st.subheader("Prediction")

    st.dataframe(
        result["data"],
        use_container_width=True
    )

    st.metric(
        "Accuracy",
        f"{result['accuracy']:.2f}%"
    )

    st.subheader(
        f"Nearest Neighbours (K = {result['neighbors']})"
    )

    st.dataframe(
        result["neighbours"],
        use_container_width=True
    )

# =========================================================
# K-MEANS
# =========================================================

elif technique == "K-Means":

    st.header("K-Means Clustering")

    result = kmeans.run()

    st.subheader("Student Cluster Assignment")

    st.dataframe(
        result["data"],
        use_container_width=True
    )

    st.subheader(
        f"Cluster Summary (K = {result['clusters']})"
    )

    st.dataframe(
        result["summary"],
        use_container_width=True
    )

# =========================================================
# APRIORI
# =========================================================

elif technique == "Apriori":

   st.title("Apriori Association Rule Mining")

   st.write("Minimum Support: 30%")
   st.write("Minimum Confidence: 50%")


   if st.button("Run Apriori"):

        try:

            result = apriori.run()

            if result.empty:
                st.warning("No association rules found.")

            else:
                st.success("Association rules generated successfully.")

                st.dataframe(
                    result,
                    use_container_width=True
                )

        except Exception as e:
            st.error(str(e))

# =========================================================
# TEXT MINING
# =========================================================

elif technique == "Text Mining":

    st.title("Text Mining")

    if st.button("Run Text Mining"):

        result = text_mining.run()

        st.subheader("Word Frequency")
        st.dataframe(
            result["frequency"],
            use_container_width=True
        )

        st.subheader("TF-IDF")
        st.dataframe(
            result["tfidf"],
            use_container_width=True
        )


# =========================================================
# WEB MINING
# =========================================================

elif technique == "Web Mining":

    st.header("🌐 Web Mining")

    st.write(
        """
        Web Mining analyzes how students use
        an online learning portal.
        """
    )

    if st.button("▶ Run Web Mining"):

        result = web_mining.run()

        st.success("Web Mining executed successfully!")

        st.subheader("🌐 Student Web Behaviour")

        st.dataframe(
            result.get("data"),
            use_container_width=True
        )

        st.subheader("📊 Behaviour Cluster Summary")

        st.dataframe(
            result.get("summary"),
            use_container_width=True
        )

        st.subheader("💡 What did we learn?")

        st.info(
            """
            Web Mining analyzes student interaction
            with an online learning system.

            Important variables include:

            • Login Count
            • Pages Visited
            • Video Watching Time
            • Quiz Attempts
            • Assignment Downloads
            """
        )


# =========================================================
# MULTIMEDIA MINING
# =========================================================

elif technique == "Multimedia Mining":

    st.header("🎥 Multimedia Mining")

    st.write(
        """
        Multimedia Mining analyzes educational
        image and video resources.
        """
    )

    if st.button("▶ Run Multimedia Mining"):

        result = multimedia_mining.run()

        st.success("Multimedia Mining executed successfully!")

        st.subheader("🎬 Multimedia Data")

        st.dataframe(
            result.get("data"),
            use_container_width=True
        )

        st.subheader("📊 Multimedia Cluster Summary")

        st.dataframe(
            result.get("summary"),
            use_container_width=True
        )

        st.subheader("💡 What did we learn?")

        st.info(
            """
            Multimedia Mining represents multimedia
            using measurable features.

            Example:

            Brightness
            Size
            Duration

            Similar resources can then be grouped together.
            """
        )


# =========================================================
# SPATIAL MINING
# =========================================================

elif technique == "Spatial Mining":

    st.header("📍 Spatial Mining")

    st.write(
        """
        Spatial Mining identifies patterns in
        area-level student data.
        """
    )

    if st.button("▶ Run Spatial Mining"):

        result = spatial_mining.run()

        st.success("Spatial Mining executed successfully!")

        st.subheader("📍 Area Data")

        st.dataframe(
            result.get("data"),
            use_container_width=True
        )

        st.subheader("📊 Area Cluster Summary")

        st.dataframe(
            result.get("summary"),
            use_container_width=True
        )

        st.subheader("💡 What did we learn?")

        st.info(
            """
            Spatial Mining studies patterns related to location.

            In this project we analyze:

            • Area
            • Latitude
            • Longitude
            • Number of Students
            • Attendance
            • Average Marks
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.write(
    "🎓 Student Data Mining Project"
)

st.sidebar.write(
    "Python + Pandas + Scikit-learn + Streamlit"
)