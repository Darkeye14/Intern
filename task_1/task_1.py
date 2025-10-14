import nltk
from newspaper import Article
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
import string

# Download required NLTK data
#pip install Newspaper3k
# pip install
def download_nltk_data():
    """Download necessary NLTK datasets if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK data (this will only happen once)...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        print("NLTK data download complete!\n")

def preprocess_text(text):
    """
    Preprocess the input text by:
    1. Converting to lowercase
{{ ... }}
    3. Removing stopwords
    4. Tokenizing into words
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        list: List of preprocessed words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize into words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return words

def score_sentences(text, words):
    """
    Score sentences based on word frequency
    
    Args:
        text (str): Original text
        words (list): Preprocessed words
        
    Returns:
        dict: Dictionary with sentences as keys and their scores as values
    """
    # Calculate word frequency
    freq_dist = FreqDist(words)
    
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Score sentences based on word frequency
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                sentence_scores[i] += freq_dist[word]
    
    return sentence_scores, sentences

def summarize_text(text, num_sentences=3):
    """
    Generate a summary of the input text
    
    Args:
        text (str): Text to summarize
        num_sentences (int): Number of sentences for the summary
        
    Returns:
        str: Generated summary
    """
    # Preprocess text
    words = preprocess_text(text)
    
    # Score sentences
    sentence_scores, sentences = score_sentences(text, words)
    
    # Get top N sentences
    top_sentences = sorted(sentence_scores.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:num_sentences]
    
    # Sort the top sentences by their original order
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    # Generate the summary
    summary = ' '.join([sentences[idx] for idx, score in top_sentences])
    
    return summary

def summarize_url(url, num_sentences=3):
    """
    Summarize an article from a URL
    
    Args:
        url (str): URL of the article to summarize
        num_sentences (int): Number of sentences for the summary
        
    Returns:
        str: Generated summary
    """
    try:
        # Initialize article
        article = Article(url)
        
        # Download and parse article
        article.download()
        article.parse()
        
        # Get article text
        text = article.text
        
        # Generate and return summary
        return summarize_text(text, num_sentences)
    except Exception as e:
        return f"Error processing URL: {str(e)}"

def main():
    # Download required NLTK data
    download_nltk_data()
    
    print("Article Summarization Tool")
    print("1. Summarize from URL")
    print("2. Summarize from text")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        # Summarize from URL
        url = input("Enter the article URL: ")
        num_sentences = int(input("Number of sentences for summary (default 3): ") or "3")
        
        print("\nGenerating summary...\n")
        summary = summarize_url(url, num_sentences)
        
    elif choice == '2':
        # Summarize from text
        print("Enter/Paste your content. Type 'END' on a new line when finished:")
        
        # Read multiple lines of input until 'END' is entered
        lines = []
        while True:
            line = input()
            if line.upper() == 'END':
                break
            lines.append(line)
            
        text = '\n'.join(lines)
        num_sentences = int(input("\nNumber of sentences for summary (default 3): ") or "3")
        
        print("\nGenerating summary...\n")
        summary = summarize_text(text, num_sentences)
    else:
        print("Invalid choice. Exiting...")
        return
    
    print("\n=== Summary ===")
    print(summary)

if __name__ == "__main__":
    main()