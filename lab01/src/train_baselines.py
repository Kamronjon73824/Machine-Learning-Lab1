import csv

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append([name, f"{accuracy:.4f}"])

    print(f"{name}: {accuracy:.4f}")


with open(
    "lab01/results/baseline_accuracy.csv",
    "w",
    newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(["model", "accuracy"])
    writer.writerows(results)