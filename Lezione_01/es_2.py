# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


# Load dataset
df = pd.read_csv("creditcard.csv")
# Assume last column is the target
X = df.drop(columns="Class")
y = df["Class"]

# Split dataset with stratify
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Decision Tree (class_weight='balanced')
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print('Decision Tree Classification Report:')
print(classification_report(y_test, dt_pred))

# 2. Random Forest (class_weight='balanced')
rf = RandomForestClassifier(n_estimators=30,class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print('Random Forest Classification Report:')
print(classification_report(y_test, rf_pred))

# ###Oversampling
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# 3. DecisionTreeClassifier with over-sampling
dt_resampled = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_resampled.fit(X_resampled, y_resampled)
dt_resampled_pred = dt_resampled.predict(X_test)
print('Decision Tree (Over-sampled) Classification Report:')
print(classification_report(y_test, dt_resampled_pred))

# 4. RandomForestClassifier with over-sampling
rf_resampled = RandomForestClassifier(n_estimators=30,class_weight='balanced', random_state=42)
rf_resampled.fit(X_resampled, y_resampled)
rf_resampled_pred = rf_resampled.predict(X_test)
print('Random Forest (Over-sampled) Classification Report:')
print(classification_report(y_test, rf_resampled_pred))

