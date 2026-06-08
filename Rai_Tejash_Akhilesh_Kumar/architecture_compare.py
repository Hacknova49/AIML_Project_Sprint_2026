import tensorflow as tf
import pandas as pd
import time
import numpy as np

class ArchitectureComparator:
    """
    A testing suite to compare Shallow, Medium, and Deep neural networks
    on different datasets, and demonstrate gradient instability (NaN loss).
    """
    
    def __init__(self):
        self.results = []
        self.datasets = self._load_datasets()

    def _load_datasets(self):
        """Loads and normalizes MNIST and Fashion-MNIST."""
        print("Loading datasets...")
        # Standard MNIST
        mnist = tf.keras.datasets.mnist
        (m_x_train, m_y_train), (m_x_test, m_y_test) = mnist.load_data()
        
        # Fashion MNIST
        f_mnist = tf.keras.datasets.fashion_mnist
        (f_x_train, f_y_train), (f_x_test, f_y_test) = f_mnist.load_data()

        return {
            'MNIST': ((m_x_train / 255.0, m_y_train), (m_x_test / 255.0, m_y_test)),
            'Fashion-MNIST': ((f_x_train / 255.0, f_y_train), (f_x_test / 255.0, f_y_test))
        }

    def get_architecture(self, name):
        """Returns uncompiled models based on the specified architecture."""
        if name == 'A_Shallow':
            return tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        elif name == 'B_Medium':
            return tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        elif name == 'C_Deep':
            return tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

    def demonstrate_nan_loss(self):
        """Intentionally causes exploding gradients, then shows the fix."""
        print("\n" + "="*50)
        print("🧪 EXPERIMENT: DELIBERATE NaN LOSS")
        print("="*50)
        
        x_train, y_train = self.datasets['MNIST'][0]
        
      
        print("1. Training with an aggressively high learning rate (100.0)...")
        broken_model = self.get_architecture('B_Medium')
        # A massive learning rate causes gradients to explode instantly
        broken_optimizer = tf.keras.optimizers.Adam(learning_rate=100.0) 
        broken_model.compile(optimizer=broken_optimizer, loss='sparse_categorical_crossentropy')
        
        history = broken_model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
        print(f"Resulting Loss: {history.history['loss'][0]} (Exploded Gradients!)")
        
        # 2. Fix the NaN Loss using proper LR and Gradient Clipping
        print("\n2. Fixing with standard LR (0.001) and Gradient Clipping (clipnorm=1.0)...")
        fixed_model = self.get_architecture('B_Medium')
        # clipnorm prevents any gradient from exceeding 1.0, ensuring mathematical stability
        fixed_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        fixed_model.compile(optimizer=fixed_optimizer, loss='sparse_categorical_crossentropy')
        
        fixed_history = fixed_model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
        print(f"Resulting Loss: {fixed_history.history['loss'][0]:.4f} (Stable and learning!)")

    def run_comparison_suite(self):
        """Trains all architectures on all datasets and logs metrics."""
        print("\n" + "="*50)
        print("🚀 RUNNING ARCHITECTURE COMPARISON SUITE")
        print("="*50)

        architectures = ['A_Shallow', 'B_Medium', 'C_Deep']
        
       
        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        )

        for dataset_name, data in self.datasets.items():
            (x_train, y_train), (x_test, y_test) = data
            
            for arch in architectures:
                print(f"\n--- Training {arch} on {dataset_name} ---")
                model = self.get_architecture(arch)
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                
                start_time = time.time()
                
           
                history = model.fit(
                    x_train, y_train,
                    epochs=20,
                    validation_split=0.2,
                    callbacks=[early_stopper],
                    verbose=0 
                )
                
                training_time = time.time() - start_time
                stopped_epoch = len(history.history['loss'])
                
                
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                
            
                self.results.append({
                    'Dataset': dataset_name,
                    'Architecture': arch,
                    'Params': model.count_params(),
                    'Epochs Run': stopped_epoch,
                    'Time (sec)': round(training_time, 2),
                    'Test Acc (%)': round(test_acc * 100, 2)
                })
                
                print(f"Finished in {stopped_epoch} epochs. Test Accuracy: {test_acc * 100:.2f}%")

    def print_final_report(self):
        """Displays a formatted DataFrame of all results."""
        print("\n" + "="*60)
        print("📊 FINAL ARCHITECTURE COMPARISON REPORT")
        print("="*60)
        df = pd.DataFrame(self.results)
        print(df.to_string(index=False))
        
        
        df.to_csv("architecture_results.csv", index=False)
        print("\n>>> Results successfully saved to 'architecture_results.csv'")



if __name__ == "__main__":
    comparator = ArchitectureComparator()
    
 
    comparator.demonstrate_nan_loss()
    
  
    comparator.run_comparison_suite()
    
   
    comparator.print_final_report()