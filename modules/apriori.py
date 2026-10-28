import pandas as pd 
from itertools import combinations 
 
def run(): 
 
    data = pd.read_csv("data/learning_activities.csv") 
 
    transactions = [ 
        set(str(x).split("|")) 
        for x in data["Activities"] 
    ] 
 
    total = len(transactions) 
    rules = [] 
 
    if total == 0: 
        return pd.DataFrame() 
 
    items = set().union(*transactions) 
 
    for a, b in combinations(items, 2): 
 
        both = sum({a, b}.issubset(t) for t in transactions) 
 
        support = both / total 
 
        if support < 0.30: 
            continue 
 
        a_count = sum(a in t for t in transactions) 
        b_count = sum(b in t for t in transactions) 
 
        for x, y, x_count, y_count in [ 
            (a, b, a_count, b_count), 
            (b, a, b_count, a_count) 
        ]: 
 
            if x_count == 0: 
                continue 
 
            confidence = both / x_count 
 
            if confidence >= 0.50: 
 
                lift = confidence / (y_count / total) 
 
                rules.append([ 
                    x, 
                    y, 
                    round(support, 2), 
                    round(confidence, 2), 
                    round(lift, 2) 
                ]) 
 
    return pd.DataFrame( 
        rules, 
        columns=[ 
            "If", 
            "Then", 
            "Support", 
            "Confidence", 
            "Lift" 
        ] 
    ).sort_values("Lift", ascending=False) 