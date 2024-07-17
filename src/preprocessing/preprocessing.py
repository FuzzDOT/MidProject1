import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class Preprocessing:
    """
    A class for text preprocessing tasks.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokens = None
    
    def handle_contractions(self, text):
        """
        Handles contractions in the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with contractions handled.
        """
        return contractions.fix(text)
    
    def clean_text(self, text):
        """
        Cleans the given text by removing non-alphabetic characters, converting to lowercase,
        and removing extra whitespaces.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        text = self.handle_contractions(text) 
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()  
        text = re.sub(r'\s+', ' ', text).strip() 
        return text
    
    def tokenize(self, text):
        """
        Tokenizes the given text into individual words.

        Args:
            text (str): The input text.

        Returns:
            list: A list of tokens (words).
        """
        return word_tokenize(text)  
    
    def remove_stop_words(self, tokens):
        """
        Removes stop words from the given list of tokens.

        Args:
            tokens (list): A list of tokens.

        Returns:
            list: A list of tokens with stop words removed.
        """
        return [word for word in tokens if word not in self.stop_words]  
    
    def lemmatize_or_stem(self, tokens):
        """
        Lemmatizes or stems the given list of tokens.

        Args:
            tokens (list): A list of tokens.

        Returns:
            list: A list of lemmatized or stemmed tokens.
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]  
    
    def preprocess(self, text):
        """
        Preprocesses the given text by applying cleaning, tokenization, stop word removal,
        and lemmatization/stemming.

        Args:
            text (str): The input text.

        Returns:
            list: A list of preprocessed tokens.
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        tokens = self.lemmatize_or_stem(tokens)
        self.tokens = tokens
        return tokens
    
    def get_tokens(self):
        """
        Returns the preprocessed tokens.

        Returns:
            list: The preprocessed tokens.
        """
        return self.tokens