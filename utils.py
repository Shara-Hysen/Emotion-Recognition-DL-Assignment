import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# =====================================================================
# 1. DATAFÖRBEREDELSE & PIPELINE
# =====================================================================


def extract_dataset(dataset):
    """
    Konverterar ett TensorFlow Dataset till vanliga NumPy-arrayer för X och y.
    """
    X, y = [], []
    
    # Loopa igenom datapaketen och spara som NumPy-format
    for images, labels in dataset:
        X.append(images.numpy())
        y.append(labels.numpy())
        
    # Slå samman listorna till sammanhängande arrayer
    return np.concatenate(X), np.concatenate(y)


# =====================================================================
# 2. TRÄNINGSUTVÄRDERING & HISTORIK
# =====================================================================


def plot_history(history, title='Träningskurvor'):
    """
    Plottar tränings- och valideringskurvor för loss och accuracy.
    """

    # Spara historiken i en DataFrame
    history_df = pd.DataFrame(history.history)

    # Skapa x-axeln (epok 1, 2, 3...)
    epochs = range(1, len(history_df) + 1)

    plt.figure(figsize=(12, 4))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_df['loss'], label='Training loss', color='#6E5B70')
    plt.plot(epochs, history_df['val_loss'], label='Validation loss', color='#C5A0A4')
    plt.xlabel('Epok')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
   
    # Subplot 2: Accuracy   
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df['accuracy'], label='Training accuracy', color='#6E5B70')
    plt.plot(epochs, history_df['val_accuracy'], label='Validation accuracy', color='#C5A0A4')
    plt.xlabel('Epok')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Layout och rubrik
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# =====================================================================
# 3. MODELLPREDIKTION & TABELLSTYLING
# =====================================================================


def predict_classes(model, X):
    """
    Gör prediktioner med en tränad modell.
    """
    # Modellens sannolikheter för varje klass
    y_proba = model.predict(X, verbose=0)

    # Välj klassen med högst sannolikhet
    y_pred = np.argmax(y_proba, axis=1)

    return y_pred, y_proba


def highlight_best_metrics(df):
    """
    Skapar en stil-matris för DataFrames som färgmarkerar de bästa nyckeltalen.
    """
    # Skapa en tom DataFrame för stilregler
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Grön styling för vinnande celler
    best_style = 'background-color: #d4edda; font-weight: bold; color: #155724'
    
    # Hitta index för bästa värdena i de tre viktigaste kolumnerna
    idx_best_acc = df['val_acc'].idxmax()
    idx_best_loss = df['best_val_loss'].idxmin()
    idx_best_epoch_acc = df['val_acc_at_best_epoch'].idxmax()
    
    # Applicera stilen på de specifika cellerna
    styles.loc[idx_best_acc, 'val_acc'] = best_style
    styles.loc[idx_best_loss, 'best_val_loss'] = best_style
    styles.loc[idx_best_epoch_acc, 'val_acc_at_best_epoch'] = best_style
    
    return styles


# =====================================================================
# 4. DJUPGÅENDE FELANALYS
# =====================================================================


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion matrix'):
    """
    Skapar och visar en confusion matrix för modellens prediktioner.
    """
    # Skapa index för klasserna
    labels = np.arange(len(class_names))

    # Beräkna confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # Skapa figur
    fig, ax = plt.subplots(figsize=(8, 6))

    # Visualisera matrisen
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap='Purples',
        xticks_rotation=90,
        values_format='d',
        colorbar=False
    )

    # Titel och layout
    plt.title(title, color='#6E5B70', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return cm


def calculate_per_class_accuracy(cm, class_names):
    """
    Beräknar accuracy och antal bilder per enskild klass.
    """
    # Hämta totalt antal bilder och antal rätt per klass
    support = cm.sum(axis=1)
    correct = np.diag(cm)

    # Beräkna accuracy och hantera division med noll
    accuracy = np.divide(
        correct,
        support,
        out=np.zeros_like(correct, dtype=float),
        where=support != 0
    )

    # Samla resultatet i en DataFrame
    result = pd.DataFrame({
        'class_id': np.arange(len(class_names)),
        'class_name': class_names,
        'support': support,
        'correct': correct,
        'accuracy': accuracy
    })

    # Sortera så att lägsta accuracy hamnar överst
    return result.sort_values('accuracy')


def get_most_confused_pairs(cm, class_names, top_n=10):
    """
    Hittar och rangordnar de klasser som modellen oftast blandar ihop.
    """
    pairs = []

    # Loopa igenom alla rader och kolumner i matrisen
    for true_class in range(cm.shape[0]):
        for predicted_class in range(cm.shape[1]):

            # Hoppa över korrekta prediktioner
            if true_class == predicted_class:
                continue

            count = cm[true_class, predicted_class]

            # Spara felaktiga prediktioner
            if count > 0:
                pairs.append({
                    'true_class': class_names[true_class],
                    'predicted_class': class_names[predicted_class],
                    'count': count
                })

    pairs_df = pd.DataFrame(pairs)

    if pairs_df.empty:
        return pairs_df

    # Sortera efter antal fel med det högsta talet överst
    return pairs_df.sort_values('count', ascending=False).head(top_n)


def plot_misclassified_examples(X, y_true, y_pred, y_proba, class_names, n_images=12, random_state=42):
    """
    Hittar, räknar och slumpar ut felklassificerade bilder för visuell granskning.
    """
    # Hitta index för alla felaktiga gissningar
    error_indices = np.where(y_true != y_pred)[0]

    # Beräkna antal och andel fel
    total = len(y_true)
    num_errors = len(error_indices)
    error_percent = (num_errors / total) * 100

    print(f'Antal felklassificerade exempel: {num_errors}')
    print(f'Andel felklassificerade: {error_percent:.2f}%')

    if num_errors == 0:
        print('Inga felklassificerade exempel att visa.')
        return

    # Slumpa ut ett bestämt antal felaktiga exempel
    rng = np.random.default_rng(random_state)
    chosen_indices = rng.choice(
        error_indices,
        size=min(n_images, len(error_indices)),
        replace=False
    )

    # Skapa rutnät baserat på valda bilder (4 kolumner)
    cols = 4
    rows = int(np.ceil(len(chosen_indices) / cols))

    plt.figure(figsize=(9, 3 * rows))

    # Rita upp bilderna i rutnätet
    for plot_index, data_index in enumerate(chosen_indices):

        true_label = y_true[data_index]
        pred_label = y_pred[data_index]
        confidence = y_proba[data_index, pred_label] * 100

        plt.subplot(rows, cols, plot_index + 1)
        plt.imshow(X[data_index].squeeze(), cmap='gray')

        plt.title(
            f'Sann: {class_names[true_label]}\n'
            f'Pred: {class_names[pred_label]}\n'
            f'Säkerhet: {confidence:.1f}%',
            fontsize=9,
            color='#6E5B70',
            y=1.0
        )
        
        plt.axis('off')
    
    plt.suptitle('Felklassificerade exempel', color='#6E5B70', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


# =====================================================================
# 5. INDIVIDUELLA BILD-PREDIKTIONER (TESTDATA & EGNA BILDER)
# =====================================================================


def visualize_prediction(image, true_label_name, y_proba_vector, class_names, top_n=5):
    """
    Skapar en visuell rapport med bilden och ett stapeldiagram över de högsta prediktionerna.
    """
    probs = y_proba_vector
    pred_label = np.argmax(probs)
    confidence = np.max(probs) * 100

    # Bestäm säkerhetsnivå
    if confidence > 70: level = 'Hög säkerhet'
    elif confidence > 40: level = 'Medel säkerhet'
    else: level = 'Låg säkerhet'

    # Hantera text och färg baserat på om sanna klassen är känd eller ej
    if true_label_name is not None:
        correct_prediction = class_names[pred_label] == true_label_name
        title_color = '#4CAF50' if correct_prediction else '#F44336'
        status_text = 'RÄTT ✓' if correct_prediction else 'FEL ✗'
        title_str = f'{status_text}\nGissad: {class_names[pred_label]} | Rätt: {true_label_name}'
    else:
        title_color = '#6E5B70'
        title_str = f'Prediktion: {class_names[pred_label]}'

    # Spara topp-gissningarna i en DataFrame
    probabilities = pd.DataFrame({'Klass': class_names, 'Sannolikhet': probs})
    probabilities = probabilities.sort_values(by='Sannolikhet', ascending=False)
    top_predictions = probabilities.head(top_n)

    # Skapa figur
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [0.7, 1.3]})

    # Visa bild
    axes[0].imshow(image.squeeze(), cmap='gray', interpolation='bicubic')
    axes[0].axis('off')
    axes[0].set_title(title_str, fontsize=13, color=title_color, fontweight='bold', pad=12)
    axes[0].text(0.5, -0.12, f'Säkerhet: {confidence:.1f}% ({level})', transform=axes[0].transAxes, ha='center', fontsize=11)

    # Sätt färger på staplarna
    bar_colors = []
    for klass in top_predictions['Klass']:
        if klass == true_label_name:
            bar_colors.append('#6E5B70')
        else:
            bar_colors.append('#DDB7AB')

    # Skapa liggande stapeldiagram
    bars = axes[1].barh(top_predictions['Klass'], top_predictions['Sannolikhet'] * 100, color=bar_colors)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Sannolikhet (%)')
    axes[1].set_title(f'Topp {top_n} prediktioner', fontsize=13, fontweight='bold', color='#6E5B70')

    # Lägg till procenttext på varje stapel
    for bar in bars:
        width = bar.get_width()
        axes[1].text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=10)

    axes[1].grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def predict_test_image(image_index, X_test, y_test, model, class_names, top_n=5):
    """
    Gör en förutsägelse på en specifik bild från testdatasetet baserat på dess index.
    """
    # Hämta bild och sann etikett
    image = X_test[image_index]
    true_label_idx = y_test[image_index]
    true_label_name = class_names[true_label_idx]
    
    # Skapa batch-dimension och predicera
    img_batch = np.expand_dims(image, axis=0)
    y_proba = model.predict(img_batch, verbose=0)[0]
    
    # Skicka till visualiseringen
    visualize_prediction(image, true_label_name, y_proba, class_names, top_n)


def predict_custom_image(image_path, model, class_names, true_label_name=None, top_n=5):
    """
    Laddar in, förbereder och förutsäger en helt extern bildfil.
    """
    # Ladda in bilden, tvinga gråskala och skala till 48x48
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    
    # Konvertera till array och normalisera
    img_array = img_to_array(img)
    img_array_normalized = img_array / 255.0
    
    # Skapa batch-dimension och predicera
    img_batch = np.expand_dims(img_array_normalized, axis=0)
    y_proba = model.predict(img_batch, verbose=0)[0]

    # Skicka till visualiseringen
    visualize_prediction(img_array_normalized, true_label_name, y_proba, class_names, top_n)


