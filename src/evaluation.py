import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    acc = accuracy_score(y_true, y_pred_classes)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred_classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred_classes)

    return acc, cm

def compare_training(histories, labels):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for i, hist in enumerate(histories):
        h = hist.history

        train_line, = axes[0].plot(
            h['loss'],
            alpha=0.5,
            label=f"{labels[i]} train"
        )

        axes[0].plot(
            h['val_loss'],
            color=train_line.get_color(),
            label=f"{labels[i]} val"
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    for i, hist in enumerate(histories):
        h = hist.history

        acc_key = 'accuracy' if 'accuracy' in h else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in h else 'val_acc'

        train_line, = axes[1].plot(
            h[acc_key],
            alpha=0.5,
            label=f"{labels[i]} train"
        )

        axes[1].plot(
            h[val_acc_key],
            color=train_line.get_color(),
            label=f"{labels[i]} val"
        )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()