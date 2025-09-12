import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import time

# Load dataset
df = pd.read_csv('Lezione_02/train.csv')

X = df.drop(columns=['label'])
y = df['label']
print(X.head())
print("Number of samples per class:")
print(y.value_counts())

# Standardize the dataframe
std= StandardScaler()
X_standardized = pd.DataFrame(
    std.fit_transform(X), 
    columns=X.columns, 
    index=X.index
)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
#Plot variance ratio for each component
X_pca = pca.fit_transform(X_standardized)
print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features after PCA: {X_pca.shape[1]}")

# 4. Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
X_standardized_train, X_standardized_test, _, _ = train_test_split(X_standardized, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Use DecisionTreeClassifier with X_train 
# Use DecisionTreeClassifier without PCA (X_standardized) and compare results
#Print time taken for training and prediction for both models
dt_pca = DecisionTreeClassifier(random_state=42)
start_pca = time.time()
dt_pca.fit(X_train, y_train)
end_pca = time.time()

print(f"Time taken for training with PCA: {end_pca - start_pca:.4f} seconds")
dt_std = DecisionTreeClassifier(random_state=42)
start_std = time.time()
dt_std.fit(X_standardized_train, y_train)
end_std = time.time()
print(f"Time taken for training without PCA: {end_std - start_std:.4f} seconds")

dt_pca_pred = dt_pca.predict(X_test)
dt_std_pred = dt_std.predict(X_standardized_test)
print('Decision Tree Classification Report with PCA:')
print(classification_report(y_test, dt_pca_pred))
print('Decision Tree Classification Report without PCA:')
print(classification_report(y_test, dt_std_pred))

"""
Con questi dati il pipeline con PCA è x3 più lento (21.06s vs 7.02s) e poco meno performante. 
Probabilmente le rotazioni delle componenti non aiutano gli split degli alberi. 
In questo caso forse meglio lavorare su iperparametri/feature selection mirata.

"""






