import os
import numpy as np
from dataset import DatasetPrepare
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


class Analyze:
    """
    A class for analyzing and visualizing text data from a dataset.

    This class includes methods for generating various plots and visualizations to understand the distribution and characteristics of text data across different categories.

    Attributes:
    - dp (DatasetPrepare): An instance of the DatasetPrepare class for loading and preparing the dataset.
    - label_id2text (dict): A dictionary mapping label ids to their corresponding text labels.
    - german_stop_words (list): A list of German stop words for text preprocessing.
    - model_path (str): The path to the pre-trained Word2Vec model.
    - plot_path (str): The directory path for saving plots.

    Methods:
    - plot_bar(x, y, x_label, y_label, title, global_average, file): Plots and saves a bar chart with the given parameters.
    - average_n_words_plot(): Plots and saves the average number of words per category.
    - n_samples_plot(): Plots and saves the number of samples per category.
    - n_unique_words_plot(): Plots and saves the number of unique words per category.
    - split_into_sentences(text): Splits a given text into sentences using regex.
    - average_n_sentence_plot(): Plots and saves the average number of sentences per category.
    - top_tfidf_words_plot(): Plots and saves the top TF-IDF words per category as word clouds.
    - preprocess_text(text): Preprocesses the given text by tokenizing and converting it to lowercase.
    - plot_tsne_embeddings(): Plots and saves a t-SNE visualization of text embeddings.
    - create_plots(): Creates all the plots defined in the class.
    """

    def __init__(self) -> None:
        self.dp = DatasetPrepare()
        self.label_id2text = {0: "Left", 1: "Center", 2: "Right"}
        self.german_stop_words = stopwords.words("german")
        self.model_path = "word2vec_model.model"
        self.plot_path = "plots"

    def plot_bar(self, x, y, x_label, y_label, title, global_average, file):
        """
        Plots and saves a bar chart.

        Parameters:
        - x (list): The x-axis values.
        - y (list): The y-axis values.
        - x_label (str): The label for the x-axis.
        - y_label (str): The label for the y-axis.
        - title (str): The title of the plot.
        - global_average (float): The global average value to be indicated on the plot.
        - file (str): The filename for saving the plot.
        """
        plt.figure(figsize=(40, 40))
        plt.bar(x, y)
        plt.xticks(fontsize=100)

        plt.yticks(
            np.arange(0, 1.5 * max(y), 10 ** np.ceil(np.log10(((1.5 * max(y)) // 10)))),
            fontsize=80,
        )
        plt.axhline(
            y=global_average,
            color="red",
            linestyle="--",
            linewidth=5,
            label="Average across all categoires",
        )
        plt.xlabel(x_label, fontsize=100)
        plt.ylabel(y_label, fontsize=100)
        plt.grid(linewidth=3)
        plt.title(title, fontsize=100)
        plt.legend(fontsize=100)
        plt.savefig(os.path.join(self.plot_path, file))

    def average_n_words_plot(self):
        """
        Plots and saves the average number of words per category.
        """
        df = self.dp.get_dataframe()
        df["word_count"] = df["text"].apply(lambda x: len(word_tokenize(str(x))))
        global_average = df["word_count"].mean()
        dfg = df.groupby("label")["word_count"].mean().reset_index()
        dfg.columns = ["label", "word_count"]
        x = [self.label_id2text[l] for l in dfg["label"].to_list()]
        y = dfg["word_count"].to_list()
        x_label = "Political bias"
        y_label = "Average number of words"
        title = "Average number of words in text per category"
        file = "average_n_words_plot"
        self.plot_bar(x, y, x_label, y_label, title, global_average, file)

    def n_samples_plot(self):
        """
        Plots and saves the number of samples per category.
        """
        df = self.dp.get_dataframe()
        global_average = len(df) / 3
        dfg = df.groupby("label").size().reset_index()
        dfg.columns = ["label", "size"]
        x = [self.label_id2text[l] for l in dfg["label"].to_list()]
        y = dfg["size"].to_list()
        x_label = "Political bias"
        y_label = "Number of samples"
        title = "Number of samples per category"
        file = "n_samples_plot"
        self.plot_bar(x, y, x_label, y_label, title, global_average, file)

    def n_unique_words_plot(self):
        """
        Plots and saves the number of unique words per category.
        """
        df = self.dp.get_dataframe()
        unique_dict = {}
        for label in self.label_id2text.keys():
            df_label = df[df["label"] == label]
            unique_dict[label] = len(
                set(word_tokenize(" ".join(df_label["text"].lower())))
            )
        x = [self.label_id2text[l] for l in unique_dict.keys()]
        y = list(unique_dict.values())
        global_average = sum(y) / 3
        x_label = "Political bias"
        y_label = "Number of unique words"
        title = "Number of unique words per category"
        file = "n_unique_words_plot"
        self.plot_bar(x, y, x_label, y_label, title, global_average, file)

    def split_into_sentences(self, text):
        """
        Splits a given text into sentences using regex.

        Parameters:
        - text (str): The input text to split.

        Returns:
        - list: A list of sentences.
        """

        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(?=\s|$)"
        sentences = re.split(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        return sentences

    def average_n_sentence_plot(self):
        """
        Plots and saves the average number of sentences per category.
        """
        df = self.dp.get_dataframe()
        df["sentence_count"] = df["text"].apply(
            lambda x: len(self.split_into_sentences(str(x)))
        )
        global_average = df["sentence_count"].mean()
        dfg = df.groupby("label")["sentence_count"].mean().reset_index()
        dfg.columns = ["label", "sentence_count"]
        x = [self.label_id2text[l] for l in dfg["label"].to_list()]
        y = dfg["sentence_count"].to_list()
        x_label = "Political bias"
        y_label = "Average number of sentences"
        title = "Average number of sentences in text per category"
        file = "average_n_sentence_plot"
        self.plot_bar(x, y, x_label, y_label, title, global_average, file)

    def top_tfidf_words_plot(self):
        """
        Plots and saves the top TF-IDF words per category as word clouds.
        """
        df = self.dp.get_dataframe()
        tfidf_vectorizer = TfidfVectorizer()
        for label in self.label_id2text.keys():
            df_label = df[df["label"] == label]
            mat = tfidf_vectorizer.fit_transform(list(df_label["text"]))
            avg_scores = mat.mean(axis=0).tolist()[0]
            terms = tfidf_vectorizer.get_feature_names_out()

            term_tfidf_scores = list(zip(terms, avg_scores))
            sorted_terms = sorted(term_tfidf_scores, key=lambda x: x[1], reverse=True)
            top_n = 500
            top_words = {
                term: score
                for term, score in sorted_terms[:top_n]
                if term not in self.german_stop_words
            }
            fig, ax = plt.subplots(figsize=(40, 40))
            wc = WordCloud(
                background_color="black",
                collocations=False,
                max_words=50,
                max_font_size=2000,
                min_font_size=8,
                width=800,
                height=1600,
                colormap=None,
            ).generate_from_frequencies(top_words)
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(
                f"Top TF-IDF Words: Importance Ranking in {self.label_id2text[label]}",
                fontsize=60,
            )
            ax.axis("off")
            plt.savefig(
                os.path.join(
                    self.plot_path,
                    f"top_tfidf_words_plot_{self.label_id2text[label]}.png",
                ),
                bbox_inches="tight",
            )

    def preprocess_text(self, text):
        """
        Preprocesses the given text by tokenizing and converting it to lowercase.

        Parameters:
        - text (str): The input text to preprocess.

        Returns:
        - list: A list of preprocessed tokens.
        """
        tokens = word_tokenize(text.lower())
        return list(tokens)

    def plot_tsne_embeddings(self):
        """
        Plots and saves a t-SNE visualization of text embeddings.
        """
        df = self.dp.get_dataframe()
        if not os.path.exists(self.model_path):
            self.dp.train_word2vec()
        model = Word2Vec.load(self.model_path).wv

        def generate_sentence_embeddings(sentence):
            return np.array(
                [model[word] for word in self.preprocess_text(sentence)]
            ).mean(axis=0)

        l = list(df["text"])
        labels = list(df["label"])
        embeds = np.array([generate_sentence_embeddings(text) for text in l])
        tsne = TSNE(n_components=2, random_state=42, perplexity=60)
        vectors_2d = tsne.fit_transform(embeds)

        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            indices = np.where(labels == label)
            plt.scatter(
                vectors_2d[indices, 0],
                vectors_2d[indices, 1],
                label=f"{self.label_id2text[label]}",
                alpha=0.5,
            )

        plt.title(
            "t-SNE Visualization of text across different categories", fontsize=35
        )
        plt.xlabel("t-SNE Component 1", fontsize=25)
        plt.ylabel("t-SNE Component 2", fontsize=25)
        plt.legend(fontsize=20)
        plt.savefig(os.path.join(self.plot_path, "tsne_plot.png"), bbox_inches="tight")

    def create_plots(self):
        self.average_n_words_plot()
        self.n_samples_plot()
        self.n_unique_words_plot()
        self.average_n_sentence_plot()
        self.top_tfidf_words_plot()
        self.plot_tsne_embeddings()


def main():
    a = Analyze()
    a.create_plots()


if __name__ == "__main__":
    main()
