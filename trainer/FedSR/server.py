
from trainer.FedAvg.server import Server as Base_Server
from utils.nets import TwinBranchNets
from .SRNet import SRNet

class Server(Base_Server):
    def __init__(self, **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = "FedSR"

        assert isinstance(self.model, TwinBranchNets), \
            "FedSR need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."

        # modify model
        self.model = SRNet(self.model, self.num_classes)

    """Server working process of FedSR is same as FedAvg"""


