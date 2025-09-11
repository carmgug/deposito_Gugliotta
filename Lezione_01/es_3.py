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

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, stratify=y, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, stratify=y_temp, test_size=0.176, random_state=42)

# Decision Tree
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)
y_pred_tree = dtree.predict(X_test)

#### Output

#Print classification report with val set
print("Validation Set Classification Report:")
print(classification_report(y_val, dtree.predict(X_val), digits=3))

print("Test Set Classification Report:")
print(classification_report(y_test, y_pred_tree, digits=3))

# Print size in % of each set 
print(f"Training set size: {len(X_train)/len(X)*100:.2f}%")
print(f"Validation set size: {len(X_val)/len(X)*100:.2f}%")
print(f"Test set size: {len(X_test)/len(X)*100:.2f}%")