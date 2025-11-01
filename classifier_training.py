import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load features and labels
fused = np.load('fused_features.npy')  # Or use pd.read_csv('fused_features.csv')
labels = pd.read_csv('labels.csv').values.ravel()  # shape should match fused features

# Split data for basic validation
X_train, X_test, y_train, y_test = train_test_split(fused, labels, test_size=0.2, stratify=labels, random_state=42)

# Train SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
