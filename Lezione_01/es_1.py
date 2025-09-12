# #### Setup iniziale
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load Dataset
df=pd.read_csv('Iris.csv')
fist_five=df.head(5)

# Split
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y= df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=42
)

# Decision Tree
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)
y_pred_tree = dtree.predict(X_test)

#### Output

# Print first five rows of the dataset
print("First five rows of the dataset:")
print(fist_five)

print("Decision Tree:")
print(classification_report(y_test, y_pred_tree, digits=3))


# Plot the decision tree
plt.figure(figsize=(12,8))
sklearn.tree.plot_tree(dtree, filled=True, feature_names=X.columns, class_names=dtree.classes_)
plt.show()