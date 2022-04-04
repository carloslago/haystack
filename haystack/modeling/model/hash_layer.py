from torch import nn


class HashLayer(nn.Module):

    def __init_subclass__(cls, **kwargs):
        """
        This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    # def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, padding_mask: torch.Tensor, **kwargs):
    #     raise NotImplementedError