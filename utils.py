import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




def plot_history(history, title='Träningskurvor'):
    """
    Plottar tränings- och valideringskurvor för loss och accuracy.
    """

    # Gör om träningshistoriken till en DataFrame för enklare hantering
    history_df = pd.DataFrame(history.history)

    # Skapar en lista med epoker (1, 2, 3, ...)
    epochs = range(1, len(history_df) + 1)

    plt.figure(figsize=(12, 4))

    # Loss-kurva
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_df["loss"], label='Training loss')
    plt.plot(epochs, history_df["val_loss"], label='Validation loss')
    plt.xlabel('Epok')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
   
    # Accuracy-kurva    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df["accuracy"], label='Training accuracy')
    plt.plot(epochs, history_df["val_accuracy"], label='Validation accuracy')
    plt.xlabel('Epok')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Huvudtitel för hela figuren
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def predict_classes(model, X):
    """
    Gör prediktioner med en tränad modell.
    """

    # Modellens sannolikheter för varje klass
    y_proba = model.predict(X, verbose=0)

    # Välj klass med högst sannolikhet
    y_pred = np.argmax(y_proba, axis=1)

    return y_pred, y_proba



def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion matrix"):
    """
    Skapar och visar en confusion matrix för modellens prediktioner.
    """

    # Skapar index för klasserna
    labels = np.arange(len(class_names))

    # Beräknar confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # Skapar figur
    fig, ax = plt.subplots(figsize=(8, 6))

    # Visualiserar confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
        colorbar=False
    )

    # Titel och layout
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cm


def calculate_per_class_accuracy(cm, class_names):

    support = cm.sum(axis=1)
    correct = np.diag(cm)

    accuracy = np.divide(
        correct,
        support,
        out=np.zeros_like(correct, dtype=float),
        where=support != 0
    )

    result = pd.DataFrame({
        "class_id": np.arange(len(class_names)),
        "class_name": class_names,
        "support": support,
        "correct": correct,
        "accuracy": accuracy
    })

    return result.sort_values("accuracy")


def get_most_confused_pairs(cm, class_names, top_n=10):
    pairs = []

    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):

            if true_class == predicted_class:
                continue

            count = cm[true_class, predicted_class]

            if count > 0:
                pairs.append({
                    "true_class": class_names[true_class],
                    "predicted_class": class_names[predicted_class],
                    "count": count
                })

    pairs_df = pd.DataFrame(pairs)

    if pairs_df.empty:
        return pairs_df

    return pairs_df.sort_values("count", ascending=False).head(top_n)


def plot_misclassified_examples(
    X,
    y_true,
    y_pred,
    y_proba,
    class_names,
    n_images=12,
    random_state=42
):

    error_indices = np.where(y_true != y_pred)[0]

    total = len(y_true)
    num_errors = len(error_indices)
    error_percent = (num_errors / total) * 100

    print(f"Antal felklassificerade exempel: {num_errors}")
    print(f"Andel felklassificerade: {error_percent:.2f}%")

    if num_errors == 0:
        print("Inga felklassificerade exempel att visa.")
        return

    rng = np.random.default_rng(random_state)

    chosen_indices = rng.choice(
        error_indices,
        size=min(n_images, len(error_indices)),
        replace=False
    )

    cols = 4
    rows = int(np.ceil(len(chosen_indices) / cols))

    plt.figure(figsize=(14, 4 * rows))

    for plot_index, data_index in enumerate(chosen_indices):

        true_label = y_true[data_index]
        pred_label = y_pred[data_index]

        confidence = y_proba[data_index, pred_label] * 100

        plt.subplot(rows, cols, plot_index + 1)

        plt.imshow(X[data_index].squeeze(), cmap="gray")

        plt.title(
            f"Sann: {class_names[true_label]}\n"
            f"Pred: {class_names[pred_label]}\n"
            f"Säkerhet: {confidence:.1f}%",
            fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

def extract_dataset(dataset):
    X, y = [], []
    for images, labels in dataset:
        X.append(images.numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

    


def show_prediction(
    image_index,
    X_data,
    y_data,
    y_proba,
    class_names,
    top_n=5
):

   
    # Hämtar bild
    image = X_data[image_index]
    true_label = y_data[image_index]

    # PREDIKTION
    probs = y_proba[image_index]

    pred_label = np.argmax(probs)

    confidence = np.max(probs) * 100

    
    # SÄKERHETSNIVÅ
    if confidence > 70:
        level = "Hög säkerhet"

    elif confidence > 40:
        level = "Medel säkerhet"

    else:
        level = "Låg säkerhet"

    
    # RÄTT / FEL
    correct_prediction = pred_label == true_label

    title_color = "green" if correct_prediction else "red"

    status_text = (
        "RÄTT ✓"
        if correct_prediction
        else "FEL ✗"
    )

    
    # SANNNOLIKHETER
    probabilities = pd.DataFrame({
        "Klass": class_names,
        "Sannolikhet": probs
    })

    probabilities = probabilities.sort_values(
        by="Sannolikhet",
        ascending=False
    )

    top_predictions = probabilities.head(top_n)


    # FIGUR
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        gridspec_kw={'width_ratios': [1, 1.2]}
    )

    
    # VISA BILD
    axes[0].imshow(
        image.squeeze(),
        cmap="gray"
    )

    axes[0].axis("off")

    axes[0].set_title(
        f"{status_text}\n"
        f"Gissad: {class_names[pred_label]} | "
        f"Rätt: {class_names[true_label]}",
        fontsize=13,
        color=title_color,
        fontweight="bold",
        pad=12
    )

    
    # SÄKERHET UNDER BILD
    axes[0].text(
        0.5,
        -0.12,
        f"Säkerhet: {confidence:.1f}% ({level})",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=11,
    )

    
    # BARFÄRGER
    bar_colors = []

    for klass in top_predictions["Klass"]:

        if klass == class_names[true_label]:
            bar_colors.append("#4CAF50")

        else:
            bar_colors.append("#CFD8DC")

    
    # BARPLOT
    bars = axes[1].barh(
        top_predictions["Klass"],
        top_predictions["Sannolikhet"] * 100,
        color=bar_colors
    )

    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Sannolikhet (%)")
    axes[1].set_title(
        f"Topp {top_n} prediktioner",
        fontsize=13,
        fontweight="bold"
    )

    # PROCENTTEXT
    for bar in bars:
        width = bar.get_width()
        axes[1].text(
            width + 1,
            bar.get_y() + bar.get_height()/2,
            f"{width:.1f}%",
            va='center',
            fontsize=10
        )


    # STIL
    axes[1].grid(
        axis='x',
        linestyle='--',
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()