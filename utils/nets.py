import copy
from typing import List
from torch import nn
from omegaconf import OmegaConf


__all__ = ['create_model', 'mlp', 'cnn', 'lstm', 'gru', 'rnn',  'twin_branch', 'TwinBranchNets']


def create_model(conf_path, num_clients, verbose=0):
    """create models by config path

    Return:
        clients_models: models list of clients
        server_model: model of server
        other_model_dict: dict of other models
    """
    conf = OmegaConf.load(conf_path)

    if verbose:
        print()
        print("=" * 30 + "{:^20}".format("Model Configs") + "=" * 30)
        print(OmegaConf.to_yaml(conf))
        print("=" * 80)
        print()

    # create client models
    models = []
    for model_conf in conf.clients_model:
        models += [eval(model_conf.name)(**model_conf.args) for _ in range(model_conf.num)]

    # use -1 client model conf create server model
    model_conf = conf.clients_model[-1]
    server_model = eval(model_conf.name)(**model_conf.args)

    # create other models
    return models[:num_clients], server_model


# --------------------
# Compose Net
# --------------------

class TwinBranchNets(nn.Module):
    """Compose feature extractor and classifier into a twin branch net.
        A twin classifier will be made for classifier.

    Args:
        feature_extractor: feature extractor, e.g. conv
        classifier: take output of f_extractor as input, and divide into classes

    Call:
        forward(self, x, use_twin=False), if use_twin, twin_classifier will be used for output.
    """

    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module, z_dim: int):
        super(TwinBranchNets, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        # Auxiliary classifier, take same structure as classifier.
        # E.g., in FedRod, this is the personalized head, while classifier is generic head.
        self.twin_classifier = copy.deepcopy(classifier)
        self.use_twin = False
        self.z_dim = z_dim

    def forward(self, x):
        feature = self.feature_extractor(x)
        x = self.classifier(feature)
        if self.use_twin:
            x += self.twin_classifier(feature)
        return x


def twin_branch(feature_extractor: OmegaConf, classifier: OmegaConf):
    z_dim = feature_extractor.args.out_dim
    fe = eval(feature_extractor.name)(**feature_extractor.args)
    cls = eval(classifier.name)(**classifier.args)
    return _twin_branch(fe, cls, z_dim)


def _twin_branch(feature_extractor: nn.Module, classifier: nn.Module, z_dim: int):
    return TwinBranchNets(feature_extractor=feature_extractor,
                          classifier=classifier,
                          z_dim=z_dim)


"""FC(MLP)"""


class FC(nn.Module):
    def __init__(self, in_features=10, out_dim=10, hidden_layers: [int] = None):
        """
         the layers num is len(hidden_layers)+1
        """
        super(FC, self).__init__()
        if hidden_layers is None:
            hidden_layers = []

        layers = []
        if len(hidden_layers) >= 1:
            in_list = [in_features] + hidden_layers
            out_list = hidden_layers + [out_dim]

            count = 0
            for in_dim, out_dim in zip(in_list, out_list):
                layers += [nn.Linear(in_features=in_dim, out_features=out_dim)]
                if count < len(hidden_layers) - 1:
                    layers += [nn.BatchNorm1d(out_dim)]
                    layers += [nn.Dropout(0.2)]
                    layers += [nn.ReLU()]
                    count += 1
        else:
            layers += [nn.Linear(in_features=in_features, out_features=out_dim, bias=True)]
        self.flatten = nn.Flatten()
        self.fcs = nn.Sequential(*layers)

    def forward(self, x):
        f = self.flatten(x)
        r = self.fcs(f)
        return r


def mlp(in_dim=10, out_dim=10, hidden_layers=None):
    return FC(in_features=in_dim,
              out_dim=out_dim,
              hidden_layers=hidden_layers)


"""
The CNN class is according to:
    'Brendan McMahan H, Moore E, Ramage D, Hampson S, Agüera y Arcas B.
    Communication-efficient learning of deep networks from decentralized data.
    Proc 20th Int Conf Artif Intell Stat AISTATS 2017. Published online 2017.
    https://arxiv.org/abs/1602.05629'
    
The Cnn2Layer, Cnn3Layer, Cnn4Layer model is according to:
    'Li D, Wang J. FedMD: Heterogenous Federated Learning via Model Distillation. 
    In: NeurIPS. ; 2019:1-8. http://arxiv.org/abs/1910.03581'
"""


def cnn(in_dim: int = 28, in_channels: int = 1, out_dim: int = 10, channels: List[int] = None):
    if channels is None:
        return CNN(in_dim=in_dim,
                   in_channels=in_channels,
                   out_dim=out_dim)
    elif len(channels) == 2:
        return Cnn2Layer(in_dim=in_dim, out_dim=out_dim, in_channels=in_channels,
                         n1=channels[0], n2=channels[1])
    elif len(channels) == 3:
        return Cnn3Layer(in_dim=in_dim, out_dim=out_dim, in_channels=in_channels,
                         n1=channels[0], n2=channels[1], n3=channels[2])
    elif len(channels) == 4:
        return Cnn4Layer(in_dim=in_dim, out_dim=out_dim, in_channels=in_channels,
                         n1=channels[0], n2=channels[1], n3=channels[2], n4=channels[3])


def _cal_out_dim(w0, kernel_size, padding, stride, pool_kernel_size=None, pool_stride=None, pool_padding=0):
    # cal according to pytorch.nn.Conv2d's doc
    w1 = int((w0 + 2 * padding - kernel_size) / stride + 1)
    # cal  according to pytorch.nn.AvgPool2d's doc
    pool_stride = pool_stride if (pool_stride is not None) else pool_kernel_size
    if pool_kernel_size is not None:
        w1 = int((w1 + 2 * pool_padding - pool_kernel_size) / pool_stride + 1)
    return w1


class CNN(nn.Module):
    """
    According to:
        'Brendan McMahan H, Moore E, Ramage D, Hampson S, Agüera y Arcas B.
        Communication-efficient learning of deep networks from decentralized data.
        Proc 20th Int Conf Artif Intell Stat AISTATS 2017. Published online 2017.
        https://arxiv.org/abs/1602.05629'

    Recommend training params:
        lr = 0.1
        Optim = SGD, According to my test, when Optim = Adam, this model can only reach 60.9% accuracy.

    """

    def __init__(self, in_dim=28, in_channels=1, out_dim=10):
        super(CNN, self).__init__()
        # ============model blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5),
                      padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=5, stride=1, padding=1,
                                    pool_kernel_size=2, pool_padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                      padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            nn.Flatten()
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=5, stride=1, padding=1,
                                    pool_kernel_size=2, pool_padding=1)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * conv_out_dim ** 2, out_features=512, bias=False),
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=out_dim, bias=False),
            # nn.ReLU(True),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Cnn4Layer(nn.Module):
    def __init__(self, in_dim=28, out_dim=10, in_channels=1, n1=64, n2=64, n3=64, n4=64, dropout_rate=0.2):
        super(Cnn4Layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=3, stride=1, padding=1),  # same padding：padding=(kernel_size-1)/2，
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=3, padding=1, stride=1, pool_kernel_size=2, pool_stride=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n1, n2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, padding=0, stride=2, pool_kernel_size=2, pool_stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(n2, n3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, padding=0, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(n3, n4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, padding=0, stride=2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n4 * conv_out_dim ** 2,
                      out_features=out_dim, bias=False),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


class Cnn3Layer(nn.Module):
    def __init__(self, in_dim=28, out_dim=10, in_channels=1, n1=128, n2=192, n3=256, dropout_rate=0.2):
        super(Cnn3Layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=(1, 1))
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=3, stride=1, padding=1,
                                    pool_kernel_size=2, pool_stride=1, pool_padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n1, n2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, stride=2, padding=0,
                                    pool_kernel_size=2, pool_stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(n2, n3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n3 * conv_out_dim ** 2, out_features=out_dim, bias=False),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class Cnn2Layer(nn.Module):
    def __init__(self, in_dim=28, out_dim=10, in_channels=1, n1=128, n2=256, dropout_rate=0.2):
        super(Cnn2Layer, self).__init__()

        # same padding：padding=(kernel_size-1)/2，
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=(1, 1))
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=3, stride=1, padding=1,
                                    pool_kernel_size=2, pool_stride=1, pool_padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n1, n2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n2 * conv_out_dim ** 2, out_features=out_dim, bias=False),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


"""
LSTM, GRU, RNN 
"""


def lstm(vocab_size=80, out_dim=80, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.2,
         bidirectional=False, tie_weights=False, embedding_path=None, seq2seq=False, use_embedding=True):
    embedding_weights = _get_embedding_weights(embedding_path)
    return RNNModel('LSTM', vocab_size, out_dim, embedding_dim, hidden_size, num_layers, dropout,
                    bidirectional, tie_weights, embedding_weights, seq2seq=seq2seq,
                    use_embedding=use_embedding)


def gru(vocab_size=80, out_dim=80, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.2,
        bidirectional=False, tie_weights=False, embedding_path=None, seq2seq=False, use_embedding=True):
    embedding_weights = _get_embedding_weights(embedding_path)
    return RNNModel('GRU', vocab_size, out_dim, embedding_dim, hidden_size, num_layers, dropout,
                    bidirectional, tie_weights, embedding_weights, seq2seq=seq2seq,
                    use_embedding=use_embedding)


def rnn(vocab_size=80, out_dim=80, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.2, bidirectional=False,
        tie_weights=False, embedding_path=None, nonlinearity='relu', seq2seq=False, use_embedding=True):
    assert nonlinearity in ['relu', 'tanh'], f"nonlinearity {nonlinearity} error"
    embedding_weights = _get_embedding_weights(embedding_path)
    return RNNModel('GRU', vocab_size, out_dim, embedding_dim, hidden_size, num_layers, dropout,
                    bidirectional, tie_weights, embedding_weights, nonlinearity, seq2seq=seq2seq,
                    use_embedding=use_embedding)


def _get_embedding_weights(embedding_path):
    # if embedding_path is None:
    #     embedding_weights = None
    # else:
    #     embedding_weights, _, _ = get_word_emb_arr(embedding_path)
    #     embedding_weights = torch.from_numpy(embedding_weights)
    return None


class RNNModel(nn.Module):
    """
    Container module with an encoder (embedding), a recurrent module (LSTM, GRU, RNN), and a decoder (FC).


    """

    def __init__(self, rnn_type, vocab_size, output_dim, embedding_dim, hidden_size, num_layers, dropout=0.2,
                 bidirectional=False,
                 tie_weights=False, embedding_weights=None, nonlinearity='relu', seq2seq=False,
                 use_embedding=True):

        super(RNNModel, self).__init__()

        # encoder
        self.use_embedding = use_embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)

        # RNN
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_size, num_layers,
                                             dropout=dropout, bidirectional=bidirectional,
                                             batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, nonlinearity=nonlinearity,
                              dropout=dropout, bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError

        if bidirectional:
            hidden_size *= 2

        # decoder
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )

        # softmax
        # self.softmax = nn.Softmax(dim=-1)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            assert hidden_size == embedding_dim, \
                'When using the tied flag, hidden_size must be equal to embedding_dim'
            self.fc.weight = self.embedding.weight

        # init weights
        if embedding_weights is not None:
            self.embedding.from_pretrained(embedding_weights)
        # init_range = 0.1
        # if embedding_weights is None:
        #     nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        # else:
        #     self.embedding.from_pretrained(embedding_weights)
        # nn.init.zeros_(self.fc.bias)
        # nn.init.uniform_(self.fc.weight, -init_range, init_range)

        self.seq2seq = seq2seq
        self.out_dim = output_dim
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        # batch first, i.e. [B, S, D]
        emb = self.embedding(x) if self.use_embedding else x

        if hidden is None:
            output, _ = self.rnn(emb)
        else:
            output, hidden = self.rnn(emb, hidden)

        if not self.seq2seq:  # final hidden state for output
            output = output[:, -1, :]

        decoded = self.fc(output)

        if self.seq2seq:
            batch_size = x.size()[0]
            decoded = decoded.view(batch_size, -1, self.out_dim)

        if hidden is None:
            return decoded  # F.log_softmax(decoded, dim=-1)  # self.softmax(decoded)
        else:
            return decoded, hidden  # F.log_softmax(decoded, dim=-1), hidden  # self.softmax(decoded), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
