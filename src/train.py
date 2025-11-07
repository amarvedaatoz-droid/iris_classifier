from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred2 = model.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example data — replace these with your own
y_true = [0, 1, 2, 2, 0, 1, 1, 0, 2, 1]
y_pred = [0, 0, 2, 2, 0, 1, 1, 1, 2, 2]

# Define class labels (replace with your actual class names)
class_labels = [ 'Setosa', 'Versicolor', 'Virginica']

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Plot and save
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.close()

print("✅ Confusion matrix saved as outputs/confusion_matrix.png")
# SAve code
import os
import joblib

# Example: assume you have a trained model (replace with your actual model)
# e.g., from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier().fit(X_train, y_train)

# Make sure you have a trained model
model = ...  # <-- your trained model object

# Ensure the outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Define the output file path
model_path = "outputs/trained_model.joblib"

# Save the model
joblib.dump(model, model_path)

print(f"✅ Model saved successfully at: {model_path}")