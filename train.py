import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import torch
from torch import nn
from models import FCLayer, transformerClassifier
from tqdm import tqdm
import logging


class TrainOnVec:
    """
    A class for training and evaluating machine learning models using vector representations of data.

    This class handles the preprocessing of data, including oversampling to balance classes, and provides methods for training and testing both a Naïve Bayes classifier and a fully connected neural network.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing text data and their corresponding vector representations and labels.
    - log_path (str): The path to the log file for logging training and testing information. Default is "logs/logs.txt".

    Attributes:
    - df (pd.DataFrame): The input dataframe.
    - X_train (np.array): The training data vectors after oversampling.
    - X_test (np.array): The testing data vectors after oversampling.
    - y_train (np.array): The training labels after oversampling.
    - y_test (np.array): The testing labels after oversampling.
    - train_indices (np.array): The indices of the training samples in the original dataframe.
    - test_indices (np.array): The indices of the testing samples in the original dataframe.
    - logger (logging.Logger): The logger for logging training and testing information.

    Methods:
    - train_bayes(): Trains a Naïve Bayes classifier on the training data.
    - test_bayes(): Tests the trained Naïve Bayes classifier on the test data and logs the classification report and error analysis.
    - train_fc(n_epochs=50, batch_size=64, lr=0.01, dropout=0.2, weight_decay=0.001): Trains a fully connected neural network on the training data.
    - test_fc(batch_size=64): Tests the trained fully connected neural network on the test data and logs the classification report and error analysis.
    """

    def __init__(self, df, log_path="logs/logs.txt") -> None:
        self.df = df
        x = np.array(list(df["vector"]), dtype=np.float16)
        y = df["label"].to_numpy(dtype=np.int16)

        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(x, y)
        resampled_indices = oversampler.sample_indices_

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.train_indices,
            self.test_indices,
        ) = train_test_split(
            X_resampled, y_resampled, resampled_indices, test_size=0.2, random_state=42
        )
        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("\n%(message)s\n")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def train_bayes(self) -> None:
        """Trains a Naïve Bayes classifier on the training data."""

        self.logger.info("Training a naïve Bayes model")
        self.nb_classifier = MultinomialNB()
        self.nb_classifier.fit(self.X_train, self.y_train)

    def test_bayes(self) -> None:
        """Tests the trained Naïve Bayes classifier on the test data and logs the classification report and error analysis."""
        y_pred = self.nb_classifier.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, digits=3)

        self.logger.info("Classification Report:")
        self.logger.info(report)

        y_pred_right = y_pred[y_pred == self.y_test]
        right_idxs = self.test_indices[y_pred == self.y_test]
        error_idxs = self.test_indices[y_pred != self.y_test]
        y_pred_error = y_pred[y_pred != self.y_test]
        texts = list(self.df["text"])
        labels = list(self.df["label"])
        self.logger.info("Error Analysis:\n")
        self.logger.info("Correct Examples:\n")
        for i in range(max(5, len(right_idxs))):
            self.logger.info(texts[right_idxs[i]])
            self.logger.info(
                f"Predicted Label: {y_pred_right[i]} - True Label: {labels[right_idxs[i]]}"
            )

        self.logger.info("Wrong Examples:\n")
        for i in range(max(5, len(error_idxs))):
            self.logger.info(texts[error_idxs[i]])
            self.logger.info(
                f"Predicted Label: {y_pred_error[i]} - True Label: {labels[error_idxs[i]]}"
            )

    def train_fc(
        self, n_epochs=50, batch_size=64, lr=0.01, dropout=0.2, weight_decay=0.001
    ) -> None:
        """Trains a fully connected neural network on the training data."""

        self.logger.info("Training a feed-forward neural network")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = self.X_train.shape[1]
        model = FCLayer(
            input_size=input_size, n_hidden=500, n_output=3, dropout=dropout
        ).to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        for _ in range(n_epochs):
            for batch_idx in tqdm(range(0, self.X_train.shape[0], batch_size)):
                x_batch = torch.tensor(
                    self.X_train[batch_idx : batch_idx + batch_size],
                    dtype=torch.float32,
                ).to(device)
                y_batch = torch.tensor(
                    self.y_train[batch_idx : batch_idx + batch_size], dtype=torch.int64
                ).to(device)
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        self.model = model

    def test_fc(self, batch_size=64) -> None:
        """Tests the trained fully connected neural network on the test data and logs the classification report and error analysis."""
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preds = []

        with torch.no_grad():
            for batch_idx in tqdm(range(0, self.X_test.shape[0], batch_size)):
                x_batch = torch.tensor(
                    self.X_test[batch_idx : batch_idx + batch_size], dtype=torch.float32
                ).to(device)
                preds.extend(self.model(x_batch).cpu().numpy().argmax(axis=-1))

        y_pred = np.array(preds)

        report = classification_report(self.y_test, y_pred, digits=3)

        self.logger.info("Classification Report:")
        self.logger.info(report)

        y_pred_right = y_pred[y_pred == self.y_test]
        right_idxs = self.test_indices[y_pred == self.y_test]
        error_idxs = self.test_indices[y_pred != self.y_test]
        y_pred_error = y_pred[y_pred != self.y_test]
        texts = list(self.df["text"])
        labels = list(self.df["label"])
        self.logger.info("Error Analysis:\n")
        self.logger.info("Correct Examples:\n")
        for i in range(max(10, len(right_idxs))):
            self.logger.info(texts[right_idxs[i]])
            self.logger.info(
                f"Predicted Label: {y_pred_right[i]} - True Label: {labels[right_idxs[i]]}"
            )

        self.logger.info("Wrong Examples:\n")
        for i in range(max(10, len(error_idxs))):
            self.logger.info(texts[error_idxs[i]])
            self.logger.info(
                f"Predicted Label: {y_pred_error[i]} - True Label: {labels[error_idxs[i]]}"
            )


class TrainTransformer:
    """
    A class for training and evaluating transformer-based classifiers.

    This class handles the preprocessing of data, including oversampling to balance classes, and provides methods for training and testing a transformer-based classifier (BERT or RoBERTa).

    Parameters:
    - df (pd.DataFrame): The input dataframe containing text data and their corresponding labels.
    - model_name (str): The name of the pre-trained transformer model to use ('bert' or 'roberta'). Default is 'bert'.
    - log_path (str): The path to the log file for logging training and testing information. Default is "logs/logs.txt".
    - plot_file (str): The path to save the training loss plot. Default is "plots/plot.png".

    Attributes:
    - device (torch.device): The device on which to run the model (GPU if available, otherwise CPU).
    - model (transformerClassifier): The transformer-based classifier model.
    - df (pd.DataFrame): The input dataframe.
    - X_train (list): The training text data after oversampling.
    - X_test (list): The testing text data after oversampling.
    - y_train (np.array): The training labels after oversampling.
    - y_test (np.array): The testing labels after oversampling.
    - train_indices (np.array): The indices of the training samples in the original dataframe.
    - test_indices (np.array): The indices of the testing samples in the original dataframe.
    - plot_file (str): The path to save the training loss plot.
    - logger (logging.Logger): The logger for logging training and testing information.

    Methods:
    - train(n_epochs=10, batch_size=32, lr=1e-4): Trains the transformer-based classifier on the training data.
    - test(batch_size=64): Tests the trained transformer-based classifier on the test data and logs the classification report and error analysis.
    """

    def __init__(
        self,
        df,
        model_name="bert",
        log_path="logs/logs.txt",
        plot_file="plots/plot.png",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = transformerClassifier(
            model_name=model_name, n_output=3, dropout=0.2
        ).to(self.device)
        self.df = df
        x = np.array((self.df["text"])).reshape(-1, 1)
        y = df["label"].to_numpy(dtype=np.int32)
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(x, y)
        resampled_indices = oversampler.sample_indices_

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.train_indices,
            self.test_indices,
        ) = train_test_split(
            X_resampled, y_resampled, resampled_indices, test_size=0.2, random_state=42
        )
        self.X_train = self.X_train.flatten().tolist()
        self.X_test = self.X_test.flatten().tolist()
        self.plot_file = plot_file
        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("\n%(message)s\n")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def train(self, n_epochs=10, batch_size=32, lr=1e-4) -> None:
        """Trains the transformer-based classifier on the training data."""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        training_loss_epochs = []
        for _ in range(n_epochs):
            training_loss = 0.0
            for batch_idx in tqdm(range(0, len(self.X_train), batch_size)):
                x_batch = self.X_train[batch_idx : batch_idx + batch_size]
                y_batch = torch.tensor(
                    self.y_train[batch_idx : batch_idx + batch_size], dtype=torch.int64
                ).to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)
                training_loss += loss.item() * len(x_batch)
                loss.backward()
                optimizer.step()

            training_loss_epochs.append(training_loss / len(self.X_train))

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, n_epochs + 1),
            training_loss_epochs,
            marker="o",
            linestyle="-",
            color="b",
            label="Training Loss",
        )

        # Add titles and labels
        plt.title("Training Loss Across Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.plot_file)

    def test(self, batch_size=64) -> None:
        """Tests the trained transformer-based classifier on the test data and logs the classification report and error analysis."""
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch_idx in tqdm(range(0, len(self.X_test), batch_size)):
                x_batch = self.X_test[batch_idx : batch_idx + batch_size]
                preds.extend(self.model(x_batch).cpu().numpy().argmax(axis=-1))

        y_pred = np.array(preds)
        report = classification_report(self.y_test, y_pred, digits=3)

        self.logger.info("Classification Report:")
        self.logger.info(report)

        y_pred_right = y_pred[y_pred == self.y_test]
        right_idxs = self.test_indices[y_pred == self.y_test]
        error_idxs = self.test_indices[y_pred != self.y_test]
        y_pred_error = y_pred[y_pred != self.y_test]
        texts = list(self.df["text"])
        labels = list(self.df["label"])
        self.logger.info("Error Analysis:\n")
        self.logger.info("Correct Examples:\n")
        for i in range(max(10, len(right_idxs))):
            self.logger.info(texts[right_idxs[i]])
            self.logger.info(
                f"Predicted Label: {y_pred_right[i]} - True Label: {labels[right_idxs[i]]}"
            )

        self.logger.info("Wrong Examples:\n")
        for i in range(max(10, len(error_idxs))):
            self.logger.info(texts[error_idxs[i]])
            self.logger.info(
                f"Predicted Label: {y_pred_error[i]} - True Label: {labels[error_idxs[i]]}"
            )
