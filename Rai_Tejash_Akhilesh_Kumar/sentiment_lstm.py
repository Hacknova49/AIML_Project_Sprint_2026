import numpy as np
import tensorflow as tf

def load_and_preprocess_data(max_features=10000, max_len=200):
    imdb = tf.keras.datasets.imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    return x_train, y_train, x_test, y_test

def build_lstm_model(max_features=10000, max_len=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=max_features, output_dim=128, input_length=max_len),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_custom_review(review, model, max_features=10000, max_len=200):
    word_index = tf.keras.datasets.imdb.get_word_index()
    words = review.lower().split()
    review_ids = []
    for word in words:
        val = word_index.get(word, 2) + 3
        if val < max_features:
            review_ids.append(val)
        else:
            review_ids.append(2)
    review_padded = tf.keras.preprocessing.sequence.pad_sequences([review_ids], maxlen=max_len)
    prediction = model.predict(review_padded, verbose=0)[0][0]
    return prediction

if __name__ == "__main__":
    max_features = 10000
    max_len = 200
    
    x_train, y_train, x_test, y_test = load_and_preprocess_data(max_features, max_len)
    
    model = build_lstm_model(max_features, max_len)
    
    model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    reviews = ["this movie was amazing and fantastic", "terrible waste of time horrible"]
    for r in reviews:
        pred = predict_custom_review(r, model, max_features, max_len)
        sentiment = "Positive" if pred > 0.5 else "Negative"
        print(f"Review: '{r}' | Score: {pred:.4f} | Sentiment: {sentiment}")


#RESULTS:-
        #Test Accuracy: 85.48%
        #Review: 'this movie was amazing and fantastic' | Score: 0.8613 | Sentiment: Positive
         #Review: 'terrible waste of time horrible' | Score: 0.0177 | Sentiment: Negative