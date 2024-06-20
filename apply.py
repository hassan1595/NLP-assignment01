from dataset import DatasetPrepare
from train import TrainOnVec, TrainTransformer


class Apply:
    """
    A class for applying various machine learning models and generating logs for text classification tasks.

    This class includes methods for generating logs for different models, including Naive Bayes, Fully Connected (FC) Neural Networks, and Transformers. It also handles the generation of PMI (Pointwise Mutual Information) words.

    Attributes:
    - label_id2text (dict): A dictionary mapping label ids to their corresponding text labels.
    - preprocess_params (dict): A dictionary containing parameters for text preprocessing.
    - vectorize_params (dict): A dictionary containing parameters for text vectorization.

    Methods:
    - get_log_name(first, preprocess_params, vectorize_params, method): Generates a log file name based on the provided parameters.
    - generate_logs_bayes(): Generates logs for Naive Bayes models with different preprocessing and vectorization parameters.
    - generate_logs_fc(): Generates logs for Fully Connected Neural Networks with different preprocessing and vectorization parameters.
    - generate_logs_transformers(): Generates logs for Transformer models (BERT and RoBERTa).
    - generate_logs_top_pmi_words(): Generates logs for the top PMI words.
    """

    def __init__(self) -> None:
        self.label_id2text = {0: "Left", 1: "Center", 2: "Right"}
        self.preprocess_params = {
            "lower": True,
            "stop_words": True,
            "lemmatize": True,
            "use_tokenizer": False,
        }

        self.vectorize_params = {
            "count_vectorizer": False,
            "tfidf": False,
            "tfidf_sublinear": False,
            "word_embed": False,
        }

    def get_log_name(self, first, preprocess_params, vectorize_params, method):
        """
        Generates a log file name based on the provided parameters.

        Parameters:
        - first (str): A prefix for the log file name.
        - preprocess_params (dict): The preprocessing parameters used.
        - vectorize_params (dict): The vectorization parameters used.
        - method (str): The method name.

        Returns:
        - str: The generated log file name.
        """
        l_all = (
            [first]
            + [p_k for p_k, p_v in preprocess_params.items() if p_v]
            + [v_k for v_k, v_v in vectorize_params.items() if v_v]
            + [method]
        )
        return "logs/" + method + "/" + "_".join(l_all) + ".txt"

    def generate_logs_bayes(self):
        """
        Generates logs for Naive Bayes models with different preprocessing and vectorization parameters.
        """
        idx = 0
        for vectorize_param_key in self.vectorize_params.keys():
            if vectorize_param_key == "word_embed":
                continue
            vectorize_params = self.vectorize_params.copy()
            vectorize_params[vectorize_param_key] = True
            dp = DatasetPrepare(**self.preprocess_params, **vectorize_params)
            df = dp.vectorize_dataframe()
            log_name = self.get_log_name(
                str(idx), self.preprocess_params, vectorize_params, "bayes"
            )
            idx += 1
            print("creating logs for ", log_name)
            t = TrainOnVec(df, log_name)
            t.train_bayes()
            t.test_bayes()
            print("finished setup")
            for (
                preprocess_param_key,
                preprocess_param_value,
            ) in self.preprocess_params.items():
                preprocess_params = self.preprocess_params.copy()
                preprocess_params[preprocess_param_key] = not preprocess_param_value
                dp = DatasetPrepare(**preprocess_params, **vectorize_params)
                df = dp.vectorize_dataframe()
                log_name = self.get_log_name(
                    str(idx), preprocess_params, vectorize_params, "bayes"
                )
                idx += 1
                print("creating logs for ", log_name)
                t = TrainOnVec(df, log_name)
                t.train_bayes()
                t.test_bayes()
                print("finished setup")

    def generate_logs_fc(self):
        """
        Generates logs for Fully Connected Neural Networks with different preprocessing and vectorization parameters.
        """

        idx = 0
        for vectorize_param_key in self.vectorize_params.keys():
            vectorize_params = self.vectorize_params.copy()
            vectorize_params[vectorize_param_key] = True
            if vectorize_param_key != "word_embed":
                vectorize_params["apply_pca"] = True
            dp = DatasetPrepare(**self.preprocess_params, **vectorize_params)
            df = dp.vectorize_dataframe()
            log_name = self.get_log_name(
                str(idx), self.preprocess_params, vectorize_params, "fc"
            )
            idx += 1
            print("creating logs for ", log_name)
            t = TrainOnVec(df, log_name)
            t.train_fc()
            t.test_fc()
            print("finished setup")
            for (
                preprocess_param_key,
                preprocess_param_value,
            ) in self.preprocess_params.items():
                preprocess_params = self.preprocess_params.copy()
                preprocess_params[preprocess_param_key] = not preprocess_param_value
                dp = DatasetPrepare(**preprocess_params, **vectorize_params)
                df = dp.vectorize_dataframe()
                log_name = self.get_log_name(
                    str(idx), preprocess_params, vectorize_params, "fc"
                )
                idx += 1
                print("creating logs for ", log_name)
                t = TrainOnVec(df, log_name)
                t.train_fc()
                t.test_fc()
                print("finished setup")

    def generate_logs_transformers(self):
        """
        Generates logs for Transformer models (BERT and RoBERTa).
        """
        dp = DatasetPrepare()
        df = dp.get_dataframe()
        t = TrainTransformer(
            df,
            model_name="bert",
            log_path="logs/transformers/bert.txt",
            plot_file="plots/bert_training.png",
        )
        t.train()
        t.test()

        dp = DatasetPrepare()
        df = dp.get_dataframe()
        t = TrainTransformer(
            df,
            model_name="roberta",
            log_path="logs/transformers/roberta.txt",
            plot_file="plots/roberta_training.png",
        )
        t.train()
        t.test()

    def generate_logs_top_pmi_words(self):
        """
        Generates logs for the top PMI (Pointwise Mutual Information) words.
        """
        d = DatasetPrepare(lemmatize=False)
        d.get_random_top_pmi(window_size=2, log_path="logs/pmi/pmi.txt")


def main():
    a = Apply()
    a.generate_logs_bayes()
    a.generate_logs_fc()
    a.generate_logs_transformers()
    a.generate_logs_top_pmi_words()


if __name__ == "__main__":
    main()
