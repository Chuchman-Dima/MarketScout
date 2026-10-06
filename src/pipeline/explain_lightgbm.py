"""
Аналіз важливості ознак та їх впливу на модель LightGBM за допомогою SHAP.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt

from common import load_train_test
from train_lightgbm import build_pipeline


def main():
    print("Завантаження даних...")
    X_train, X_test, y_train_raw, _ = load_train_test()
    y_train = np.log1p(y_train_raw)

    print("Навчання пайплайну LightGBM...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    lgbm_model = pipe.named_steps["model"]

    print("Трансформація тестової вибірки...")
    X_test_transformed = pipe.named_steps["categorize"].transform(X_test)

    # Для прискорення розрахунків беремо випадкову підвибірку (наприклад, 2000 авто)
    X_sample = X_test_transformed.sample(n=2000, random_state=42)

    print("Розрахунок SHAP-значень...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer(X_sample)

    # Summary plot (Beeswarm)
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("../../models_results/shap_beeswarm_lightgbm.png", dpi=300)
    plt.close()

    # Bar plot (середня абсолютна важливість фічей)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("../../models_results/shap_bar_lightgbm.png", dpi=300)
    plt.close()

    print("Графіки збережено у папку models_results: shap_beeswarm_lightgbm.png, shap_bar_lightgbm.png")


if __name__ == "__main__":
    main()