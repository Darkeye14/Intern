import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import random
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)

class TextGenerator:
    def __init__(self, model_path='text_generator_model.h5', tokenizer_path='tokenizer.pkl', seq_length=50):
        """
        Initialize the Text Generator with optional pre-trained model and tokenizer.
        
        Args:
            model_path (str): Path to save/load the model
            tokenizer_path (str): Path to save/load the tokenizer
            seq_length (int): Length of input sequences
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.seq_length = seq_length
        self.tokenizer = None
        self.model = None
        self.vocab_size = 0
        self.max_sequence_len = 100  # Maximum length for generated text
        
        # Try to load existing model and tokenizer
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            self.load_model_and_tokenizer()
    
    def build_model(self, vocab_size, embedding_dim=100, lstm_units=256):
        """Build the LSTM model architecture."""
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=self.seq_length),
            LSTM(lstm_units, return_sequences=True),
            Dropout(0.2),
            LSTM(lstm_units),
            Dropout(0.2),
            Dense(lstm_units, activation='relu'),
            Dense(vocab_size, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer if they exist."""
        try:
            self.model = load_model(self.model_path)
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.vocab_size = len(self.tokenizer.word_index) + 1
            print("Loaded pre-trained model and tokenizer.")
            return True
        except Exception as e:
            print(f"Could not load model/tokenizer: {e}")
            return False
    
    def save_model_and_tokenizer(self):
        """Save the current model and tokenizer."""
        self.model.save(self.model_path)
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Model and tokenizer saved to {self.model_path} and {self.tokenizer_path}")
    
    def train_on_text(self, text_blocks, epochs=20, batch_size=128):
        """
        Train the model on a list of text blocks.
        
        Args:
            text_blocks (list): List of text strings to train on
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        if not text_blocks:
            print("No text data provided for training.")
            return
            
        # Tokenize the text
        if not self.tokenizer:
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(text_blocks)
            self.vocab_size = len(self.tokenizer.word_index) + 1
        
        # Create sequences of tokens
        sequences = []
        for text in text_blocks:
            tokens = self.tokenizer.texts_to_sequences([text])[0]
            for i in range(self.seq_length, len(tokens)):
                seq = tokens[i-self.seq_length:i+1]
                sequences.append(seq)
        
        if not sequences:
            print("Not enough data to create training sequences.")
            return
        
        sequences = np.array(sequences)
        X, y = sequences[:, :-1], sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)
        
        # Build or load model
        if not self.model:
            self.model = self.build_model(self.vocab_size)
        
        # Train the model
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.save_model_and_tokenizer()
    
    def generate_text(self, seed_text, num_words=100, temperature=0.7):
        """
        Generate text based on a seed text.
        
        Args:
            seed_text (str): Starting text for generation
            num_words (int): Number of words to generate
            temperature (float): Controls randomness in generation (0.1-1.0)
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            return "Error: Model or tokenizer not loaded. Please train or load a model first."
        
        # Clean and prepare seed text
        seed_text = seed_text.lower()
        seed_text = re.sub(r'[^\w\s]', '', seed_text)
        
        # Generate text word by word
        result = seed_text
        for _ in range(num_words):
            # Tokenize the seed text
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            # Pad the sequence
            token_list = pad_sequences([token_list], maxlen=self.seq_length, padding='pre')
            # Predict the next word
            predicted_probs = self.model.predict(token_list, verbose=0)[0]
            
            # Apply temperature
            predicted_probs = np.log(predicted_probs) / temperature
            exp_probs = np.exp(predicted_probs)
            predicted_probs = exp_probs / np.sum(exp_probs)
            
            # Sample from the distribution
            predicted_id = np.random.choice(len(predicted_probs), p=predicted_probs)
            
            # Map back to word
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_id:
                    output_word = word
                    break
            
            # Add to result and update seed text
            result += " " + output_word
            seed_text += " " + output_word
            seed_text = ' '.join(seed_text.split()[-self.seq_length:])
            
            # Stop if we hit a period (end of sentence)
            if output_word.endswith('.'):
                if len(result.split()) >= num_words // 2:  # Ensure minimum length
                    break
        
        # Post-process the generated text
        sentences = sent_tokenize(result)
        if sentences:
            # Capitalize first letter of first sentence
            sentences[0] = sentences[0][0].upper() + sentences[0][1:]
            # Ensure proper spacing after punctuation
            result = ' '.join(sentences)
            result = re.sub(r'\s+([.,!?;:])', r'\1', result)  # Remove space before punctuation
            result = re.sub(r'([.,!?;:])(\w)', lambda m: m.group(1) + ' ' + m.group(2).lower(), result)  # Add space after punctuation
        
        return result

def load_local_training_data(filepath='training_data.txt'):
    """Load training data from a local text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split into paragraphs and clean up
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return paragraphs
    except Exception as e:
        print(f"Error loading training data: {e}")
        return []

def main():
    print("=== LSTM Text Generator ===")
    print("This program generates coherent paragraphs on any topic using LSTM.")
    
    # Initialize the text generator
    text_gen = TextGenerator()
    
    # If no pre-trained model exists, load training data and train a new one
    if not text_gen.model or not text_gen.tokenizer:
        print("No pre-trained model found. Training a new one...")
        print("Loading training data from local file...")
        
        # Load training data from local file
        training_data = load_local_training_data('training_data.txt')
        
        if not training_data:
            print("Error: Could not load training data. Please make sure 'training_data.txt' exists.")
            return
        
        # Train the model
        print("\nTraining the model (this may take a while)...")
        text_gen.train_on_text(training_data, epochs=20, batch_size=128)
        print("\nModel trained successfully!")
    
    # Interactive text generation
    print("\nEnter a topic or starting text (or 'quit' to exit):")
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input:
            print("Please enter a topic or starting text.")
            continue
        
        # Generate text
        print("\nGenerating text...\n")
        generated_text = text_gen.generate_text(
            seed_text=user_input,
            num_words=150,  # Generate about a paragraph
            temperature=0.7  # Controls randomness (0.1-1.0)
        )
        
        print("Generated Text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50 + "\n")
        print("Enter another topic or 'quit' to exit.")

if __name__ == "__main__":
    main()