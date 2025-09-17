# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("creditcard.csv")
# Split
X = df.drop(columns="Class")
y = df["Class"]


# K-Fold stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
auc_tree = cross_val_score(dt, X, y, cv=skf, scoring="roc_auc")

print(f"Decision Tree AUC: {auc_tree.mean():.3f} ± {auc_tree.std():.3f}")
