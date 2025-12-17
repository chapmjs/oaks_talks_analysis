"""
Text processing utilities for General Conference talks analysis.
"""

import re
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from utils.custom_stopwords import get_stopwords

# Download required NLTK data
def setup_nltk():
    """Download required NLTK data packages."""
    nltk_downloads = [
        'punkt', 'stopwords', 'wordnet', 
        'averaged_perceptron_tagger', 'omw-1.4'
    ]
    for package in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

# Initialize NLTK
setup_nltk()

class TextProcessor:
    def __init__(self, use_lemmatization=True):
        """
        Initialize text processor.
        
        Args:
            use_lemmatization: Whether to lemmatize words
        """
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stopwords = get_stopwords(include_religious=True)
        
    def clean_text(self, text):
        """
        Basic text cleaning.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers that stand alone
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_talk_content(self, filepath):
        """
        Extract just the talk content from a saved file.
        
        Args:
            filepath: Path to talk file
            
        Returns:
            Talk content text
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip the header metadata
        if '\n---\n' in content:
            content = content.split('\n---\n', 1)[1]
        
        return content
    
    def tokenize(self, text, lowercase=True):
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            lowercase: Whether to convert to lowercase
            
        Returns:
            List of tokens
        """
        if lowercase:
            text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation tokens
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove tokens that are just punctuation
        tokens = [token for token in tokens if any(c.isalpha() for c in token)]
        
        return tokens
    
    def remove_stopwords(self, tokens, custom_stopwords=None):
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            custom_stopwords: Additional stopwords to remove
            
        Returns:
            Filtered token list
        """
        stopwords = self.stopwords.copy()
        if custom_stopwords:
            stopwords.update(custom_stopwords)
        
        return [token for token in tokens if token.lower() not in stopwords]
    
    def get_wordnet_pos(self, treebank_tag):
        """
        Convert treebank POS tag to wordnet POS tag.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens using POS tagging.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.lemmatizer:
            return tokens
        
        # Get POS tags
        pos_tags = nltk.pos_tag(tokens)
        
        # Lemmatize with POS
        lemmatized = []
        for token, pos in pos_tags:
            wordnet_pos = self.get_wordnet_pos(pos)
            lemma = self.lemmatizer.lemmatize(token, pos=wordnet_pos)
            lemmatized.append(lemma)
        
        return lemmatized
    
    def process_text(self, text, remove_stops=True, lemmatize=True):
        """
        Full text processing pipeline.
        
        Args:
            text: Raw text
            remove_stops: Whether to remove stopwords
            lemmatize: Whether to lemmatize
            
        Returns:
            Processed tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text, lowercase=True)
        
        # Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize and self.lemmatizer:
            tokens = self.lemmatize_tokens(tokens)
        
        return tokens
    
    def get_word_frequencies(self, tokens, n_most_common=None):
        """
        Get word frequency counts.
        
        Args:
            tokens: List of tokens
            n_most_common: Number of most common words to return
            
        Returns:
            Counter object or list of (word, count) tuples
        """
        counter = Counter(tokens)
        
        if n_most_common:
            return counter.most_common(n_most_common)
        
        return counter
    
    def extract_ngrams(self, tokens, n=2):
        """
        Extract n-grams from token list.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-gram tuples
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def get_sentences(self, text):
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

def combine_talk_texts(talk_files):
    """
    Combine multiple talk files into single text.
    
    Args:
        talk_files: List of file paths
        
    Returns:
        Combined text string
    """
    processor = TextProcessor()
    combined = []
    
    for filepath in talk_files:
        content = processor.extract_talk_content(filepath)
        combined.append(content)
    
    return '\n\n'.join(combined)
