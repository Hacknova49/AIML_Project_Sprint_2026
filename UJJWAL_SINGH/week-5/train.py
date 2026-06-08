import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(67)
tf.random.set_seed(67)



def load_csv_rows(path):
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset CSV not found at: {path}")
    df = pd.read_csv(path, header=0, low_memory=False)
    if df.shape[1] < 2:
        raise ValueError(f"CSV at {path} does not look like MNIST/Fashion-MNIST format")
    y = df.iloc[:, 0].astype('int32').values
    # Pixel values are 0-255, scale them to 0.0-1.0
    X = df.iloc[:, 1:].astype('float32').values / 255.0
    return X, y


def get_architectures(input_dim=784, num_classes=10):
    arch_a = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ], name="Architecture_A_Shallow")

    arch_b = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ], name="Architecture_B_Medium")

    arch_c = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ], name="Architecture_C_Deep")

    return {
        "Shallow (A)": arch_a,
        "Medium (B)": arch_b,
        "Deep (C)": arch_c
    }

def plot_curves(history, title, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    ax1.plot(epochs, history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax1.plot(epochs, history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history.history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history.history:
        ax2.plot(epochs, history.history['val_accuracy'], label='Val Acc')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    datasets = ["fashion_mnist"]
    comparison_results = []

    for dataset_name in datasets:
        
        print(f"Processing Dataset: {dataset_name.upper()}")
        

        train_path = "./fashion-mnist_train.csv"
        test_path = "./fashion-mnist_test.csv"

        try:
            X_train, y_train = load_csv_rows(train_path)
            X_test, y_test = load_csv_rows(test_path)

        except FileNotFoundError as e:
            print(e)
            return

        # Split a validation set from train
        val_split = 0.2
        split_idx = int(len(X_train) * (1 - val_split))
        # X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_train_sub, y_train_sub = X_train[:split_idx], y_train[:split_idx]

        architectures = get_architectures(input_dim=X_train.shape[1])


        print("\n\nRunning Overfitting Test (Without EarlyStopping)")
        overfit_histories = {}
        for name, model_orig in architectures.items():
            print(f"Training {name} (Overfitting)...")
            # Clone model to avoid shared weights
            model = tf.keras.models.clone_model(model_orig)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            start_time = time.time()
            history = model.fit(
                X_train_sub, y_train_sub,
                # validation_data=(X_val, y_val),
                validation_split=0.2,
                epochs=20,
                batch_size=128,
                verbose=0
            )
            elapsed = time.time() - start_time
            overfit_histories[name] = history

            # Save curves
            plot_curves(history, f"{dataset_name.upper()} {name} (Overfitting)", output_dir / f"{dataset_name}_{name.replace(' ', '_').lower()}_overfit.png")



        print("\n--- Running with EarlyStopping ---")
        early_stop_histories = {}
        early_stop_epochs = {}
        
        for name, model_orig in architectures.items():
            print(f"Training {name} with EarlyStopping...")
            model = tf.keras.models.clone_model(model_orig)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                restore_best_weights=True, 
                verbose=1)
            
            start_time = time.time()
            history = model.fit(
                X_train_sub, y_train_sub,
                # validation_data=(X_val, y_val),
                validation_split=0.2,
                epochs=20,
                batch_size=128,
                callbacks=[early_stop],
                verbose=0
            )
            elapsed = time.time() - start_time
            epochs_trained = len(history.history['loss'])
            
            early_stop_histories[name] = history
            early_stop_epochs[name] = epochs_trained
            
            # Evaluate on test set
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            num_params = model.count_params()

            comparison_results.append({
                "Dataset": dataset_name,
                "Architecture": name,
                "Complexity (Params)": num_params,
                "Stopped Epoch": epochs_trained,
                "Accuracy (EarlyStopped)": test_acc,
                "Test Loss (EarlyStopped)": test_loss,
                "Training Time (s)": elapsed,
                "Avg Time per Epoch (s)": elapsed / epochs_trained
            })

            # Save classification report
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            report = classification_report(y_test, y_pred)
            with open(output_dir / f"{dataset_name}_{name.replace(' ', '_').lower()}_report.txt", "w") as f:
                f.write(report)

            # Save curves
            plot_curves(history, f"{dataset_name.upper()} {name} (EarlyStopping)", output_dir / f"{dataset_name}_{name.replace(' ', '_').lower()}_earlystop.png")



        print("\n--- Simulating and Fixing NaN Loss ---")
        # We will use Architecture B for this demonstration
        nan_model = tf.keras.models.clone_model(architectures["Medium (B)"])
        
        # High learning rate to trigger NaN (unstable weight updates)
        bad_optimizer = tf.keras.optimizers.Adam(learning_rate=1e10)
        nan_model.compile(optimizer=bad_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print("Training with huge learning rate (1e10) to cause NaN loss...")
        nan_history = nan_model.fit(
            X_train_sub[:1000], y_train_sub[:1000],  # Small subset for quick demonstration
            epochs=3,
            batch_size=128,
            verbose=1
        )
        print("Loss values in NaN simulation epochs:", nan_history.history['loss'])

        print("FixingNaN loss: compiling with a stable learning rate (1e-3) and re-running...")
        fixed_model = tf.keras.models.clone_model(architectures["Medium (B)"])
        good_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        fixed_model.compile(optimizer=good_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        fixed_history = fixed_model.fit(
            X_train_sub[:1000], y_train_sub[:1000],
            epochs=3,
            batch_size=128,
            verbose=1
        )
        print("Loss values after fixing:", fixed_history.history['loss'])


    df_compare = pd.DataFrame(comparison_results)
    df_compare.to_csv(output_dir / "architecture_comparison.csv", index=False)
    print("\n\n\nFINAL COMPARISON ")
    print(df_compare.to_string(index=False))
    
    print(f"Results and plots successfully saved to: {output_dir}")

if __name__ == "__main__":
    main()
