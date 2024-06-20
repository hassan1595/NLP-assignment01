import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class FCLayer(nn.Module):
    """
    A fully connected neural network layer with one hidden layer, ReLU activation, and optional dropout.

    Parameters:
    - input_size (int): The size of the input features.
    - n_hidden (int): The number of neurons in the hidden layer. Default is 500.
    - n_output (int): The number of output neurons, typically the number of classes for classification. Default is 3.
    - dropout (float): The dropout probability. Default is 0.0 (no dropout).

    Attributes:
    - l_1 (nn.Linear): The first linear layer (input to hidden).
    - l_2 (nn.Linear): The second linear layer (hidden to output).
    - relu (nn.ReLU): ReLU activation function.
    - dropout (nn.Dropout): Dropout layer.

    Methods:
    - forward(x): Defines the forward pass through the network. Applies the first linear layer, followed by ReLU activation,
                then dropout, and finally the second linear layer to produce the output.
    """

    def __init__(self, input_size, n_hidden=500, n_output=3, dropout=0.0):
        super().__init__()
        self.l_1 = nn.Linear(input_size, n_hidden)
        self.l_2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x,
    ):
        x = self.relu(self.l_1(x))
        x = self.dropout(x)
        x = self.l_2(x)

        return x


class transformerClassifier(nn.Module):
    """
    A transformer-based classifier using BERT or RoBERTa as the base model with an additional fully connected classification head.

    Parameters:
    - model_name (str): The name of the pre-trained transformer model to use. Either 'bert' or 'roberta'. Default is 'bert'.
    - n_hidden (int): The number of neurons in the hidden layer of the classification head. Default is 500.
    - n_output (int): The number of output neurons, typically the number of classes for classification. Default is 3.
    - dropout (float): The dropout probability in the classification head. Default is 0.0 (no dropout).
    - device (torch.device): The device on which to run the model. Default is GPU if available, otherwise CPU.

    Attributes:
    - tokenizer (AutoTokenizer): The tokenizer corresponding to the selected pre-trained model.
    - model (AutoModel): The pre-trained transformer model.
    - device (torch.device): The device on which the model is run.
    - l_1 (nn.Linear): The first linear layer in the classification head (from transformer output to hidden).
    - l_2 (nn.Linear): The second linear layer in the classification head (from hidden to output).
    - relu (nn.ReLU): ReLU activation function.
    - dropout (nn.Dropout): Dropout layer.

    Methods:
    - forward(texts): Defines the forward pass through the model. Tokenizes the input texts, passes them through the transformer model,
                      applies the classification head (linear layer, ReLU, dropout, and final linear layer), and produces the output.
    """

    def __init__(
        self,
        model_name="bert",
        n_hidden=500,
        n_output=3,
        dropout=0.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        if model_name == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google-bert/bert-base-german-cased"
            )
            self.model = AutoModel.from_pretrained("google-bert/bert-base-german-cased")
        elif model_name == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "fusionbase/deu-business-registry-roberta-base"
            )
            self.model = AutoModel.from_pretrained(
                "fusionbase/deu-business-registry-roberta-base"
            )
        else:
            raise ValueError("only bert and distilbert are supported")

        self.device = device
        self.l_1 = nn.Linear(768, n_hidden)
        self.l_2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, texts):
        x = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)
        x = self.model(x["input_ids"])["pooler_output"]
        x = self.relu(self.l_1(x))
        x = self.dropout(x)
        x = self.l_2(x)

        return x
