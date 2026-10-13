import csv
import os
import threading
import time
import warnings
from pathlib import Path
from statistics import median

import joblib
import psutil

from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------
# Paths
# --------------------------------------------------

RESULTS_DIR = Path("lab01") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# Models
# --------------------------------------------------

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

# --------------------------------------------------
# Memory measurement
# --------------------------------------------------

process = psutil.Process(os.getpid())


def measure_peak_memory(function):
    peak_memory = process.memory_info().rss
    stop_event = threading.Event()

    def monitor():
        nonlocal peak_memory

        while not stop_event.is_set():
            current_memory = process.memory_info().rss

            if current_memory > peak_memory:
                peak_memory = current_memory

            time.sleep(0.001)

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    try:
        result = function()

    finally:
        stop_event.set()
        monitor_thread.join()

        current_memory = process.memory_info().rss

        if current_memory > peak_memory:
            peak_memory = current_memory

    return result, peak_memory


# --------------------------------------------------
# Measurements
# --------------------------------------------------

results = []

for name, base_model in models.items():

    print()
    print(f"Measuring {name}...")

    # ----------------------------------------------
    # Warm-up training
    # ----------------------------------------------

    warmup_model = clone(base_model)
    warmup_model.fit(X_train, y_train)

    # ----------------------------------------------
    # Training time
    # Median of 5 runs after warm-up
    # ----------------------------------------------

    training_times = []

    for _ in range(5):

        model = clone(base_model)

        start = time.perf_counter()

        model.fit(X_train, y_train)

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000

        training_times.append(elapsed_ms)

    median_training_ms = median(training_times)

    # ----------------------------------------------
    # Peak training memory
    # ----------------------------------------------

    memory_model = clone(base_model)

    _, peak_training_bytes = measure_peak_memory(
        lambda: memory_model.fit(X_train, y_train)
    )

    peak_training_mb = peak_training_bytes / (1024 ** 2)

    # ----------------------------------------------
    # Final trained model
    # ----------------------------------------------

    final_model = clone(base_model)
    final_model.fit(X_train, y_train)

    sample = X_test[0].reshape(1, -1)

    # ----------------------------------------------
    # Inference warm-up
    # ----------------------------------------------

    final_model.predict(sample)

    # ----------------------------------------------
    # Single-sample inference
    # 100 repetitions, report median
    # ----------------------------------------------

    inference_times = []

    for _ in range(100):

        start = time.perf_counter()

        final_model.predict(sample)

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000

        inference_times.append(elapsed_ms)

    median_inference_ms = median(inference_times)

    # ----------------------------------------------
    # Peak inference memory
    # ----------------------------------------------

    def inference_memory_test():

        for _ in range(100):
            final_model.predict(sample)

    _, peak_inference_bytes = measure_peak_memory(
        inference_memory_test
    )

    peak_inference_mb = peak_inference_bytes / (1024 ** 2)

    # ----------------------------------------------
    # Save model and calculate model size
    # ----------------------------------------------

    filename = f"{name}.joblib"

    model_path = RESULTS_DIR / filename

    joblib.dump(
        final_model,
        model_path
    )

    model_size_bytes = model_path.stat().st_size

    model_size_kb = model_size_bytes / 1024

    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    results.append([
        name,
        f"{median_training_ms:.4f}",
        f"{median_inference_ms:.4f}",
        model_size_bytes,
        f"{model_size_kb:.2f}",
        f"{peak_training_mb:.2f}",
        f"{peak_inference_mb:.2f}"
    ])

    # ----------------------------------------------
    # Print results
    # ----------------------------------------------

    print(
        f"Training median: "
        f"{median_training_ms:.4f} ms"
    )

    print(
        f"Inference median: "
        f"{median_inference_ms:.4f} ms"
    )

    print(
        f"Model size: "
        f"{model_size_kb:.2f} KB"
    )

    print(
        f"Peak training memory: "
        f"{peak_training_mb:.2f} MB"
    )

    print(
        f"Peak inference memory: "
        f"{peak_inference_mb:.2f} MB"
    )

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

output_file = RESULTS_DIR / "system_measurements.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "model",
        "training_median_ms",
        "inference_median_ms",
        "model_size_bytes",
        "model_size_kb",
        "peak_training_memory_mb",
        "peak_inference_memory_mb"
    ])

    writer.writerows(results)

print()
print(
    "Saved to "
    "lab01/results/system_measurements.csv"
)