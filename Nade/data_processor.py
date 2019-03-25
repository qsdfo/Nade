from DatasetManager.music_dataset import MusicDataset
from torch import nn

from Nade.helpers import cuda_variable


class DataProcessor(nn.Module):
    def __init__(self, dataset: MusicDataset, embedding_dim):
        self.embedding_dim = embedding_dim
        self.dataset = dataset
        super(DataProcessor, self).__init__()

    def embed(self, x):
        """
        flatten and embed

        :param x: any
        :return: (batch_size, sequence_length, embedding_dim)
        """
        raise NotImplementedError

    def mask(self, x, p=None):
        raise NotImplementedError

    @staticmethod
    def mean_crossentropy(pred, target):
        """

        :param pred: any
        :param target: any
        :return:
        """
        raise NotImplementedError

    def pred_seq_to_preds(self, pred_seq):
        """

        :param pred_seq: (batch_size, length, num_features)
        :return:
        """
        raise NotImplementedError

    def local_position(self, batch_size, sequence_length):
        raise NotImplementedError

    def get_relative_attention_module(self, *args):
        raise NotImplementedError

    def preprocessing(self, *tensors):
        return [cuda_variable(t.long())
                for t in tensors]

    def get_len_max_seq(self):
        raise NotImplementedError
