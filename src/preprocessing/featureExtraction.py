import numpy as np
from gensim.models import Word2Vec

class WordEmbeddings_Handler:
    """
    A class to handle Word2Vec word embeddings.

    Attributes
    ----------
    vector_size : int, default=100
        Dimensionality of the word vectors.
    window : int, default=5
        Maximum distance between the current and predicted word within a sentence.
    min_count : int, default=1
        Ignores all words with a total frequency lower than this.
    workers : int, default=4
        Number of worker threads to train the model.
    model : gensim.models.Word2Vec or None
        Word2Vec model instance.

    """
    
    def __init__(self, vector_size=50, window=5, min_count=1, workers=4, epochs=50):
        """
        Initialize a WordEmbeddings_Handler instance.
        
        Parameters
        ----------
        vector_size : int, default=100
            Dimensionality of the word vectors.
        window : int, default=5
            Maximum distance between the current and predicted word within a sentence.
        min_count : int, default=1
            Ignores all words with a total frequency lower than this.
        workers : int, default=4
            Number of worker threads to train the model.
        epochs : int, default=20
            Number of iterations (epochs) over the corpus.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def build_model(self, data):
        """
        Build the Word2Vec model vocabulary from input data.
        
        Parameters
        ----------
        data : iterable of iterables
            Input data for training the Word2Vec model.
        """
        self.model = Word2Vec(vector_size=self.vector_size,
                              window=self.window,
                              min_count=self.min_count,
                              workers=self.workers
                             )
        self.model.build_vocab(data)
    
    def train_model(self, data, epochs=20):
        """
        Train the Word2Vec model on input data.
        
        Parameters
        ----------
        data : iterable of iterables
            Input data for training the Word2Vec model.
        epochs : int, default=20
            Number of iterations (epochs) over the corpus.
        """
        self.model.train(data, total_examples=len(data), epochs=epochs)
    
    def get_embeddings(self, word):
        """
        Get the vector representation of a word from the trained Word2Vec model.
        
        Parameters
        ----------
        word : str
            Word to retrieve embeddings for.

        Returns
        -------
        numpy.ndarray or None
            Vector representation of the word if found, None otherwise.
        """
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            return None
        
    def infer_matrix(self, data):
        """
        Generate an embeddings matrix for a collection of sentences.
        
        Parameters
        ----------
        data : iterable of iterables
            Input data where each element is a list of tokens (words).

        Returns
        -------
        numpy.ndarray
            Matrix of embeddings for each sentence in data.
        """
        embeddings_matrix = []

        for tokens in data:
            sentence_embedding = []

            for token in tokens:
                embedding = self.get_embeddings(token)
                if embedding is not None:
                    sentence_embedding.append(embedding)

            if sentence_embedding:
                sentence_embedding = np.mean(sentence_embedding, axis=0)
            else:
                sentence_embedding = np.zeros(self.vector_size)

            embeddings_matrix.append(sentence_embedding)

        return np.array(embeddings_matrix)