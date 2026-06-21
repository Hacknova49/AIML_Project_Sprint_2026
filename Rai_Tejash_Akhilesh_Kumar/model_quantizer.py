import tensorflow as tf
import numpy as np
import os

def generate_dummy_data():
    x = np.random.rand(500, 28, 28, 1).astype(np.float32)
    y = np.random.randint(0, 10, size=(500,))
    return x, y

def build_base_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def representative_dataset_gen():
    x_val, _ = generate_dummy_data()
    for i in range(100):
        yield [np.expand_dims(x_val[i], axis=0)]

def main():
    x_train, y_train = generate_dummy_data()
    model = build_base_model()
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter_float.convert()
    
    with open("model_float32.tflite", "wb") as f:
        f.write(tflite_float_model)
        
    converter_int = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int.representative_dataset = representative_dataset_gen
    converter_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int.inference_input_type = tf.int8
    converter_int.inference_output_type = tf.int8
    tflite_int8_model = converter_int.convert()
    
    with open("model_int8.tflite", "wb") as f:
        f.write(tflite_int8_model)
        
    size_float = os.path.getsize("model_float32.tflite") / 1024
    size_int = os.path.getsize("model_int8.tflite") / 1024
    
    print(f"Float32 Model Size: {size_float:.2f} KB")
    print(f"INT8 Model Size: {size_int:.2f} KB")
    print(f"Compression Ratio: {size_float/size_int:.2f}x")

if __name__ == "__main__":
    main()