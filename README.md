# 🎓 Student Intelligence and Learning Analytics System

A Python-based data mining project that explores student academic performance, learning activities, feedback, web usage, multimedia data, and spatial patterns through an interactive Streamlit application.

## 📌 Project Overview

The Student Intelligence and Learning Analytics System brings multiple data mining techniques together in one interface. Users can select a technique from the sidebar and explore the corresponding results, prediction tables, model metrics, or cluster summaries.

## 🚀 Live Demo

**[Launch the Student Data Mining System](https://student-data-mining-system-uxy7znvvvqx98a47adzgbt.streamlit.app/)**

Explore the interactive dashboard and try the student data mining and learning analytics techniques directly in your browser. No local installation required.

## ✨ Features

- **Classification:** ID3-style entropy-based decision tree, J48-style decision tree, Naive Bayes, Decision Tree, and K-Nearest Neighbours (K-NN).
- **Clustering:** K-Means for grouping similar student records.
- **Association Rule Mining:** Apriori for discovering relationships in data.
- **Text Mining:** Word frequency analysis and TF-IDF.
- **Web Mining:** Analysis of student activity on an online learning platform.
- **Multimedia Mining:** Exploration and grouping of multimedia-related features.
- **Spatial Mining:** Analysis of location-related and area-level patterns.
- **Interactive Dashboard:** A Streamlit interface for selecting techniques and viewing results.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 📁 Project Structure

```text
Students_Data_Mining_Project/
├── data/
│   ├── learning_activities.csv
│   ├── multimedia_data.csv
│   ├── spatial_data.csv
│   ├── student_data.csv
│   ├── student_feedback.txt
│   ├── Student_ID_Activities.csv
│   └── web_usage.csv
├── modules/
│   ├── apriori.py
│   ├── decision_tree.py
│   ├── id3.py
│   ├── j48.py
│   ├── kmeans.py
│   ├── knn.py
│   ├── multimedia_mining.py
│   ├── naive_bayes.py
│   ├── spatial_mining.py
│   ├── text_mining.py
│   └── web_mining.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/halakhan2007/Student-Data-Mining-System.git
cd Student-Data-Mining-System
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can access the dashboard.

## 📊 Student Performance Classification

The classification modules use student attributes such as:

- Study hours
- Attendance
- Assignment scores
- Internal marks
- Previous scores

The ID3-style module uses an entropy-based decision tree to predict student results and reports accuracy, feature importance, and the root attribute.

## 🎯 Project Objectives

- Explore different data mining techniques using educational datasets.
- Examine patterns in student performance and learning behaviour.
- Compare classification approaches.
- Explore student groupings and association patterns.
- Present analysis results through an interactive dashboard.

## 👩‍💻 Project Status

Developed as an academic Data Science project to practice Python programming, data analysis, machine learning, and interactive application development.
