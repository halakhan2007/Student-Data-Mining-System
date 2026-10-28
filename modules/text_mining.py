import pandas as pd 
from collections import Counter 
from sklearn.feature_extraction.text import TfidfVectorizer 
 
def run(): 
 
    with open("data/student_feedback.txt", encoding="utf-8") as f: 
        docs = [x.lower() for x in f] 
 
    stop = {"the", "is", "are", "a", "an", "to", "of", "and", "for"} 
 
    words = [ 
        w.strip(".,!?") 
        for d in docs 
        for w in d.split() 
        if w.lower() not in stop 
    ] 
 
    frequency = pd.DataFrame( 
        Counter(words).most_common(), 
        columns=["Word", "Frequency"] 
    ) 
 
    vectorizer = TfidfVectorizer(stop_words=list(stop)) 
    matrix = vectorizer.fit_transform(docs) 
 
    tfidf = pd.DataFrame({ 
        "Word": vectorizer.get_feature_names_out(), 
        "TF_IDF": matrix.mean(axis=0).A1 
    }).sort_values("TF_IDF", ascending=False) 
 
    return { 
        "frequency": frequency, 
        "tfidf": tfidf 
    } 