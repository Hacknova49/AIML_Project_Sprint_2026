import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_dummy_dataset():
    np.random.seed(42)
    
    x_train = np.random.rand(400, 32, 32, 3).astype('float32')
    y_train = np.random.randint(0, 4, size=(400,))
    
    x_test = np.random.rand(100, 32, 32, 3).astype('float32')
    y_test = np.random.randint(0, 4, size=(100,))
    
    y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_test = tf.keras.utils.to_categorical(y_test, 4)
    
    return x_train, y_train, x_test, y_test

def build_transfer_learning_model(input_shape=(32, 32, 3), num_classes=4):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Transfer Learning Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('transfer_learning_metrics.png')
    plt.close()

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = generate_dummy_dataset()
    
    model = build_transfer_learning_model(input_shape=(32, 32, 3), num_classes=4)
    
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2
    )
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy*100:.2f}%")
    
    plot_training_history(history)