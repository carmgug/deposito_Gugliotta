import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import time
from sklearn.metrics import confusion_matrix
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
pca = PCA(n_components=0.95,whiten=True)  # Retain 95% of variance
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

#Print confusion matrix
cm_pca = confusion_matrix(y_test, dt_pca_pred)
cm_std = confusion_matrix(y_test, dt_std_pred)
print("Confusion Matrix with PCA:")
print(cm_pca)
print("Confusion Matrix without PCA:")
print(cm_std)

#Print 3d with PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Use PCA to reduce to 3 components for visualization
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_standardized)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis', s=5)
ax.set_title('3D PCA of the Dataset')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Informazioni sulla varianza spiegata
total_variance = sum(pca_3d.explained_variance_ratio_)
print(f"Varianza spiegata dalle prime 3 componenti: {total_variance:.3f}")
for i, var in enumerate(pca_3d.explained_variance_ratio_):
    print(f"Componente {i+1}: {var:.3f}")
plt.tight_layout()
plt.show()



"""
Con questi dati il pipeline con PCA è x3 più lento (21.06s vs 7.02s) e poco meno performante. 
Probabilmente le rotazioni delle componenti non aiutano gli split degli alberi. 
In questo caso forse meglio lavorare su iperparametri/feature selection mirata.

"""








