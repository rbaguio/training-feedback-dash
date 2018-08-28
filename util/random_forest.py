import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

url = 'https://docs.google.com/spreadsheets/d/1onA4xqBUa_uXDQB6qcS5p-dCetGj9Kpgo3D-VKl1XLw/export?gid=1988319498&format=csv'

df = pd.read_csv(url)
df.drop(df.columns[-2:], inplace=True, axis=1)

df.columns = [
    "participant_name",
    "role",
    "course",
    "instructor-name",
    "instructor-clarity",
    "instructor-brevity",
    "instructor-quality",
    "instructor-enthusiasm",
    "course-content",
    "course-organization",
    "course-amount-learned",
    "course-relevance",
    "comment-most-like",
    "comment-least-like",
    "comment-improvement",
    "net-promoter-score"
]

X = df[df.columns[4:12]]
y = df['net-promoter-score']

forest = ExtraTreesClassifier(
    n_estimators=10,
    max_depth=8,
    max_leaf_nodes=128
)

forest.fit(X, y)

features = X.columns

importances = forest.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in forest.estimators_],
    axis=0
)

# Reverse sort the indices
indices = np.argsort(importances)[::-1]

series = pd.Series({
    feature: score for feature, score in zip(
        features, std
    )
})
