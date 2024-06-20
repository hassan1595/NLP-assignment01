import math
import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
import re
import spacy
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import defaultdict
import logging


class DatasetPrepare:
    """
    This class is responsible for preparing a dataframe by implementing various preprocessing and vectorization
    pipelines. Multiple preprocessing pipelines can be used simultaneously, but only one vectorization method
    is allowed at a time.

    Parameters:
    - lower (bool): If True, converts all text to lowercase.
    - stop_words (bool): If True, removes stop words from the text.
    - lemmatize (bool): If True, lemmatizes the text.
    - use_tokenizer (bool): If True, tokenizes the text using a specified tokenizer.
    - count_vectorizer (bool): If True, uses Count Vectorization for vectorizing the text.
    - tfidf (bool): If True, uses TF-IDF Vectorization for vectorizing the text.
    - tfidf_sublinear (bool): If True, applies sublinear TF scaling in TF-IDF Vectorization.
    - word_embed (bool): If True, uses word embeddings for vectorizing the text.
    - apply_pca (bool): If True, applies Principal Component Analysis (PCA) for dimensionality reduction.
    - pca_dim (int): The number of dimensions to reduce the data to if PCA is applied.

    Note:
    - While multiple preprocessing steps can be applied simultaneously, only one vectorization method
    (count_vectorizer, tfidf, tfidf_sublinear, or word_embed) should be selected.
    """

    def __init__(
        self,
        lower=True,
        stop_words=True,
        lemmatize=True,
        use_tokenizer=False,
        count_vectorizer=True,
        tfidf=False,
        tfidf_sublinear=False,
        word_embed=False,
        apply_pca=False,
        pca_dim=200,
    ) -> None:
        self.dataset_path = os.path.join("datasets", "political_bias_dataset.csv")
        self.model_path = "word2vec_model.model"

        self.preprocess_params = {
            "lower": lower,
            "stop_words": stop_words,
            "lemmatize": lemmatize,
            "use_tokenizer": use_tokenizer,
        }

        self.vectorize_params = {
            "count_vectorizer": count_vectorizer,
            "tfidf": tfidf,
            "tfidf_sublinear": tfidf_sublinear,
            "word_embed": word_embed,
        }
        self.extra_params = {
            "apply_pca": apply_pca,
            "pca_dim": pca_dim,
        }

        for attr, value in self.preprocess_params.items():
            setattr(self, attr, value)

        for attr, value in self.vectorize_params.items():
            setattr(self, attr, value)

        for attr, value in self.extra_params.items():
            setattr(self, attr, value)

    def get_dataframe(self) -> pd.DataFrame:
        """
        returns the raw dataframe
        """

        def normalize_whitespace(text):
            # Strip leading and trailing whitespace
            text = text.strip()
            # Replace multiple whitespace characters with a single space
            text = re.sub(r"\s+", " ", text)
            return text

        df = pd.read_csv(self.dataset_path, index_col=0)
        df["text"] = df["title"] + " " + df["body"]
        df["text"] = df["text"].apply(normalize_whitespace)
        df["text"] = df["text"].apply(lambda x: re.sub(r"[^\w\s]", "", x))

        return df

    def get_random_top_pmi(self, window_size=2, log_path="logs/logs.txt"):
        """
        This function finds the most similar words for the top 100 words from the most important 20000 words
        according to TF-IDF scores. Due to computational limits, only the top 20000 most important words are considered.
        For each of these top 100 words, the most similar words are determined using Pointwise Mutual Information (PMI).

        Parameters:
        - window_size (int): The size of the context window to consider for co-occurrences. Default is 2.
        - log_path (str): The path to the log file where the most similar words for each of the selected words will be logged. Default is "logs/logs.txt".

        Steps:
        1. Preprocess the dataframe and tokenize the text.
        2. Compute TF-IDF scores and select the top 20000 words based on their importance.
        3. Clean tokens by retaining only the selected top 20000 words.
        4. Calculate co-occurrence counts within the specified window size.
        5. Select the top 100 words from the vocabulary based on TF-IDF scores.
        6. For each of these top 100 words, compute the PMI with all other words and log the most similar words.
        """

        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("\n%(message)s\n")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        df = self.preprocess_dataframe()
        df["tokens"] = df["preprocessed"].apply(word_tokenize)
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=word_tokenize)
        mat = tfidf_vectorizer.fit_transform(list(df["preprocessed"]))
        avg_scores = mat.mean(axis=0).tolist()[0]
        terms = tfidf_vectorizer.get_feature_names_out()
        term_tfidf_scores = list(zip(terms, avg_scores))
        sorted_terms = sorted(term_tfidf_scores, key=lambda x: x[1], reverse=True)
        selected_vocab = [v for v, _ in sorted_terms[:20000]]

        tokens_clean = []
        for tokens in tqdm(df["tokens"], "removing unimportant tokens"):
            tokens_clean.append([token for token in tokens if token in selected_vocab])

        co_occurrence = defaultdict(lambda: defaultdict(int))
        for tokens in tokens_clean:
            for i, token in enumerate(tokens):
                for j in range(
                    max(i - window_size, 0), min(i + window_size + 1, len(tokens))
                ):
                    if i != j:
                        co_occurrence[token][tokens[j]] += 1

        vocab = list(co_occurrence.keys())
        # idxs = np.random.randint(0, len(vocab), size = (100,))
        # vocab_selected = [vocab[idx] for idx in idxs]
        vocab_selected = selected_vocab[:100]
        print(vocab_selected)

        total_occurrences = sum(sum(d.values()) for d in co_occurrence.values())

        def calculate_pmi(word1, word2):
            p_word1 = sum(co_occurrence[word1].values()) / total_occurrences
            p_word2 = sum(co_occurrence[word2].values()) / total_occurrences
            p_word1_word2 = co_occurrence[word1][word2] / total_occurrences
            if p_word1_word2 > 0:
                pmi = math.log(p_word1_word2 / (p_word1 * p_word2))
            else:
                pmi = 0
            return pmi

        for word in vocab_selected:
            pmi_l = []
            for other_word in tqdm(vocab):
                if word == other_word:
                    pmi_l.append(0)
                else:
                    pmi_l.append(calculate_pmi(word, other_word))

            best_idxs = np.array(pmi_l).argsort()[::-1][:100]
            best_pmi_vcab = [vocab[idx] for idx in best_idxs]

            self.logger.info(f"Most similar word for {word} -> {best_pmi_vcab[0]} ")

        return co_occurrence

    def train_word2vec(self):
        """
        Trains a word2vec model on the dataset
        """

        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            return tokens

        df = self.get_dataframe()
        preprocessed_corpus = [preprocess_text(doc) for doc in list(df["text"])]
        model = Word2Vec(
            sentences=preprocessed_corpus,
            vector_size=200,
            window=5,
            min_count=1,
            workers=20,
            negative=20,
            epochs=20,
        )
        model.save(self.model_path)
        return model

    def preprocess_dataframe(self) -> pd.DataFrame:
        """
        Applies the preprocessing pipeline
        """

        df_name = "df_data"
        for artb_name, atrb in self.preprocess_params.items():
            if atrb and artb_name != "use_tokenizer":
                df_name += "_" + artb_name
        df_name += ".csv"
        df_path = os.path.join("cache", df_name)

        if os.path.exists(df_path):
            print("found in cache")
            df = pd.read_csv(df_path, index_col=0)
            return df

        df = self.get_dataframe()
        df["preprocessed"] = df["text"].copy()

        if self.lower:
            df["preprocessed"] = df["preprocessed"].apply(lambda x: x.lower())
        if self.stop_words:
            german_stop_words = stopwords.words("german")
            df["preprocessed"] = df["preprocessed"].apply(
                lambda x: " ".join(
                    [
                        word
                        for word in x.split()
                        if word.lower() not in german_stop_words
                    ]
                )
            )

        if self.lemmatize:
            nlp = spacy.load("de_core_news_md")

            def lemmatize_text_german(text):
                """
                Lemmatize all words in the given German text.
                """
                doc = nlp(text)
                return " ".join(
                    [x.lemma_.lower() if self.lower else x.lemma_ for x in doc]
                )

            df["preprocessed"] = df["preprocessed"].apply(lemmatize_text_german)

        df.to_csv(df_path)
        return df

    def vectorize_dataframe(self) -> pd.DataFrame:
        """
        Applies the vectorization pipeline
        """

        n_true = sum(list(self.vectorize_params.values()))
        if n_true != 1:
            raise ValueError("Exactly only one vectorisation method is allwoed")

        df = self.preprocess_dataframe()

        if self.count_vectorizer:
            vectorizer = CountVectorizer(
                lowercase=False, tokenizer=word_tokenize if self.use_tokenizer else None
            )
            vec = vectorizer.fit_transform(list(df["preprocessed"]))
            df["vector"] = list(vec.toarray())

        if self.tfidf or self.tfidf_sublinear:
            vectorizer = TfidfVectorizer(
                lowercase=False,
                sublinear_tf=self.tfidf_sublinear,
                tokenizer=word_tokenize if self.use_tokenizer else None,
            )
            vec = vectorizer.fit_transform(list(df["preprocessed"]))
            df["vector"] = list(vec.toarray())

        if self.word_embed:
            if os.path.exists(self.model_path):
                model = Word2Vec.load(self.model_path).wv
            else:
                model = self.train_word2vec()

            def generate_sentence_embeddings(tokens):
                return list(
                    np.array([model[token] for token in tokens if token in model]).mean(
                        axis=0
                    )
                )

            df["tokens"] = df["preprocessed"].apply(word_tokenize)
            df["vector"] = df["tokens"].apply(generate_sentence_embeddings)

        if self.apply_pca:
            x = np.array(list(df["vector"]), dtype=np.float32)
            pca = PCA(n_components=self.pca_dim)
            df["vector"] = list(pca.fit_transform(x))

        return df
