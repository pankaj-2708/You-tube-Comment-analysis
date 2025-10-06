import mlflow
import pandas as pd
from pathlib import Path
import yaml
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_deps(model_name, model_version):
    mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def generate_report(output_path, input_path, model_name, model_version):

    # Load model
    model = load_deps(model_name, model_version)

    # Load data
    df = pd.read_csv(input_path)
    test_y = df["Sentiment"]
    test_X = df.drop(columns=["Sentiment"])

    # Predictions
    prob = model.predict(test_X).values
    y_pred = prob.argmax(axis=1)

    # Metrics
    acc = accuracy_score(test_y, y_pred)
    pre = precision_score(test_y, y_pred, average=None)
    rec = recall_score(test_y, y_pred, average=None)
    f1 = f1_score(test_y, y_pred, average=None)

    # Confusion Matrix
    conf_mat = confusion_matrix(test_y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    conf_mat_path = output_path / "conf_mat.png"
    plt.savefig(conf_mat_path)
    plt.close()

    # ROC-AUC
    auc_scores = roc_auc_score(test_y, prob, multi_class="ovr", average=None)

    fpr_list, tpr_list, labels = [], [], ["Negative", "Neutral", "Positive"]
    for i in range(prob.shape[1]):
        fpr, tpr, _ = roc_curve([1 if j == i else 0 for j in test_y], prob[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    # ROC Curve Plot
    plt.figure(figsize=(6, 5))
    for i in range(len(labels)):
        plt.plot(fpr_list[i], tpr_list[i], label=f"{labels[i]} (AUC={auc_scores[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OvR)")
    plt.legend()
    roc_path = output_path / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # Generate Markdown dynamically
    metrics_table = "| Metric | " + " | ".join(labels) + " |\n"
    metrics_table += "|--------|" + "|".join(["--------"] * len(labels)) + "|\n"
    metrics_table += f"| Precision | " + " | ".join([f"{p:.3f}" for p in pre]) + " |\n"
    metrics_table += f"| Recall    | " + " | ".join([f"{r:.3f}" for r in rec]) + " |\n"
    metrics_table += f"| F1-score  | " + " | ".join([f"{f:.3f}" for f in f1]) + " |\n"

    auc_list_md = "\n".join([f"- {labels[i]}: {auc_scores[i]:.3f}" for i in range(len(labels))])

    report_md = f"""
# Model Evaluation Report

**Model:** `{model_name}`  
**Version:** `{model_version}`  

---

## 1. Overview
This report evaluates the model performance on the test dataset. Metrics, confusion matrix, and ROC curves are provided for multiclass sentiment prediction.

---

## 2. Performance Metrics

{metrics_table}

**Overall Accuracy:** {acc:.3f}  

**ROC-AUC (OvR):**  
{auc_list_md}

---

## 3. Confusion Matrix

![Confusion Matrix]({conf_mat_path.name})

---

## 4. ROC Curve

![ROC Curve]({roc_path.name})

---
"""

    report_path = output_path / "model_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)


def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    output_path = home_dir / "reports"
    input_path = home_dir / "data" / "train_test_split" / "test.csv"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["model_report"]

    if not params["generate_report"]:
        return

    generate_report(
        output_path, input_path, params["best_model_name"], params["best_model_version"]
    )


if __name__ == "__main__":
    main()
