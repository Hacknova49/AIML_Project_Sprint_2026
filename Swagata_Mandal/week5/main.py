
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv("train42000.csv")

X = train_df.iloc[:, 1:].values.astype("float32") / 255.0
y = train_df.iloc[:, 0].values

y_cat = to_categorical(y, 10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

plt.figure(figsize=(12,6))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap="gray")
    plt.title(np.argmax(y_train[i]))
    plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sample_images.png", dpi=300, bbox_inches="tight")
plt.close()

def shallow_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def medium_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def deep_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

def train_and_evaluate(model):
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - start
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return history, accuracy, loss, training_time

model_A = shallow_model()
history_A, acc_A, loss_A, time_A = train_and_evaluate(model_A)

model_B = medium_model()
history_B, acc_B, loss_B, time_B = train_and_evaluate(model_B)

model_C = deep_model()
history_C, acc_C, loss_C, time_C = train_and_evaluate(model_C)

plt.figure(figsize=(10,6))
plt.plot(history_A.history['val_accuracy'], label='Shallow')
plt.plot(history_B.history['val_accuracy'], label='Medium')
plt.plot(history_C.history['val_accuracy'], label='Deep')
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10,6))
plt.plot(history_A.history['val_loss'], label='Shallow')
plt.plot(history_B.history['val_loss'], label='Medium')
plt.plot(history_C.history['val_loss'], label='Deep')
plt.legend()
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

models = ["Shallow","Medium","Deep"]
accuracies = [acc_A*100, acc_B*100, acc_C*100]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies)
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

times = [time_A, time_B, time_C]

plt.figure(figsize=(8,5))
plt.bar(models, times)
plt.savefig(f"{OUTPUT_DIR}/training_time_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

params = [model_A.count_params(), model_B.count_params(), model_C.count_params()]

plt.figure(figsize=(8,5))
plt.bar(models, params)
plt.savefig(f"{OUTPUT_DIR}/parameter_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

predictions = model_C.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

report = classification_report(y_true, y_pred)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

results = pd.DataFrame({
    "Architecture": models,
    "Accuracy (%)": [round(acc_A*100,2), round(acc_B*100,2), round(acc_C*100,2)],
    "Loss": [round(loss_A,4), round(loss_B,4), round(loss_C,4)],
    "Training Time (s)": [round(time_A,2), round(time_B,2), round(time_C,2)],
    "Parameters": params
})

results.to_csv(f"{OUTPUT_DIR}/final_results.csv", index=False)
results.to_excel(f"{OUTPUT_DIR}/final_results.xlsx", index=False)

best_model = models[np.argmax(accuracies)]

summary = f"""
Architecture Comparison Project Results

Best Model: {best_model}

{results.to_string(index=False)}
"""

with open(f"{OUTPUT_DIR}/project_summary.txt", "w") as f:
    f.write(summary)

print(results)
print("All outputs saved to:", OUTPUT_DIR)
