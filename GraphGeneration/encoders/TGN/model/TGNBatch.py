import torch

class TGNBatch:
    def __init__(self, **kwargs):
        self.__set_attributes(kwargs)

    def __set_attributes(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self