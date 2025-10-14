import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import brown
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

# Ensure reproducibility (to some extent)
np.random.seed(42)

def download_nltk_data():
    """Download required NLTK corpora if missing."""
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def load_corpus():
    """Load and return a reasonably sized text corpus as a single string."""
    # Use Brown corpus (smaller than Gutenberg, faster to prepare)
    sentences = brown.sents()
    # Join tokens into sentences, then into one large text
    joined_sentences = [' '.join(tokens) for tokens in sentences]
    corpus_text = '\n'.join(joined_sentences)
    # Basic cleanup: keep letters, numbers, punctuation
    corpus_text = re.sub(r"\s+", " ", corpus_text).strip()
    return corpus_text

def build_tokenizer(corpus_text, vocab_size=20000):
    """Fit a Keras Tokenizer on text and return it along with total vocab size."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts([corpus_text])
    # Actual vocab used may be <= vocab_size + 1 due to OOV token
    vocab_len = min(vocab_size, len(tokenizer.word_index)) + 1
    return tokenizer, vocab_len

def make_sequences(tokenizer, corpus_text, seq_length=20):
    """Create (X, y) supervised sequences for language modeling."""
    # Convert entire corpus to a single sequence of token IDs
    token_list = tokenizer.texts_to_sequences([corpus_text])[0]
    sequences = []
    for i in range(seq_length, len(token_list)):
        seq = token_list[i - seq_length:i + 1]
        sequences.append(seq)
    sequences = np.array(sequences, dtype=np.int32)
    X, y = sequences[:, :-1], sequences[:, -1]
    return X, y

def build_model(vocab_size, seq_length=20, embed_dim=128, lstm_units=256):
    """Build a simple LSTM language model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_length),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001)
    )
    return model

def sample_with_temperature(preds, temperature=0.7, top_k=50, exclude_ids=None):
    """Sample an index from a probability array using temperature and top-k, excluding given ids."""
    preds = np.asarray(preds).astype('float64')
    exclude_ids = set(exclude_ids or [])

    # Zero out excluded indices
    for idx in exclude_ids:
        if 0 <= idx < preds.shape[0]:
            preds[idx] = 0.0

    # If all probabilities are zero (after exclusion), fallback to uniform over non-excluded
    if not np.any(preds > 0):
        preds = np.ones_like(preds, dtype='float64')
        for idx in exclude_ids:
            if 0 <= idx < preds.shape[0]:
                preds[idx] = 0.0

    # Select top-k indices
    if top_k is not None and 0 < top_k < preds.shape[0]:
        nonzero = np.flatnonzero(preds)
        k = min(top_k, nonzero.size) if nonzero.size > 0 else 0
        top_indices = nonzero if k == 0 else np.argpartition(preds, -k)[-k:]
    else:
        top_indices = np.flatnonzero(preds)

    if top_indices.size == 0:
        return int(np.argmax(preds))

    top_probs = preds[top_indices]
    # Temperature scaling
    top_probs = np.log(np.maximum(top_probs, 1e-12)) / max(temperature, 1e-6)
    top_probs = np.exp(top_probs)
    top_probs = top_probs / np.sum(top_probs)

    choice = np.random.choice(len(top_indices), p=top_probs)
    return int(top_indices[choice])

def generate_text(model, tokenizer, seed_text, num_words=120, seq_length=20, temperature=0.7, top_k=50):
    """Generate text conditioned on a seed string."""
    result_words = seed_text.strip().split()

    # Build exclusion set (padding id 0 and OOV id if present)
    oov_id = tokenizer.word_index.get("<OOV>")
    exclude_ids = {0}
    if oov_id is not None:
        exclude_ids.add(oov_id)

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([' '.join(result_words[-seq_length:])])[0]
        seq = pad_sequences([seq], maxlen=seq_length, padding='pre')
        preds = model.predict(seq, verbose=0)[0]

        next_id = sample_with_temperature(preds, temperature=temperature, top_k=top_k, exclude_ids=exclude_ids)
        word = tokenizer.index_word.get(next_id, None)

        if (word is None) or (word == "<OOV>"):
            # Fallback to highest-prob non-excluded valid token
            for idx in np.argsort(preds)[::-1]:
                if int(idx) not in exclude_ids and tokenizer.index_word.get(int(idx), None):
                    next_id = int(idx)
                    word = tokenizer.index_word[next_id]
                    break
        if word is None:
            # As a last resort, skip this token
            continue
        result_words.append(word)

    # Post-process to one paragraph
    text = ' '.join(result_words)
    # Simple spacing fixes
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text

def generate_gpt2_paragraph(topic: str, num_words: int = 120,
                            model_name: str = "gpt2",
                            temperature: float = 0.7,
                            top_p: float = 0.9,
                            top_k: int = 50,
                            repetition_penalty: float = 1.2,
                            no_repeat_ngram_size: int = 3) -> str:
    """Generate a coherent paragraph on a topic using GPT-2 (no training)."""
    set_seed(42)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    prompt = (
        f"Write a coherent, well-structured single paragraph about {topic}. "
        f"Use clear, natural language and avoid lists."
    )

    # Rough token estimate from word target
    max_new_tokens = max(40, min(256, int(num_words * 1.5)))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # Keep only the continuation beyond the prompt
    generated = full_text[len(prompt):].strip()
    # Ensure single paragraph formatting
    generated = re.sub(r"\s+", " ", generated).strip()
    # End with terminal punctuation
    if generated and generated[-1] not in ".!?":
        generated += "."
    return f"{prompt} {generated}"

def main():
    print("=== GPT-2 Paragraph Generator (Topic-Conditioned) ===")
    topic = input("Enter a topic: ").strip() or "technology"
    try:
        num_words = int(input("Approx. number of words to generate (default 120): ") or 120)
    except ValueError:
        num_words = 120

    print("\nGenerating paragraph with GPT-2 (first run will download the model)...\n")
    paragraph = generate_gpt2_paragraph(
        topic=topic,
        num_words=num_words,
        model_name="gpt2",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    print("=== Generated Paragraph ===\n")
    print(paragraph)

    os.makedirs('output', exist_ok=True)
    out_path = os.path.join('output', 'task_4_generated.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(paragraph)
    print(f"\nSaved generated paragraph to: {out_path}")

if __name__ == "__main__":
    main()
