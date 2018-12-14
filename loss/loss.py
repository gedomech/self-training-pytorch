import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, reduce=True, size_average=True):
        super().__init__()
        weight = torch.Tensor(weight)

        self.loss = nn.NLLLoss(weight, reduce=reduce, size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class JensenShannonDivergence(nn.Module):

    def __init__(self, reduce=True, size_average=False):
        super().__init__()

        self.loss = nn.KLDivLoss(reduce=reduce, size_average=size_average)

    def forward(self, ensemble_probs):
        n, c, h, w = ensemble_probs.shape  # number of distributions
        mixture_dist = ensemble_probs.mean(0, keepdim=True).expand(n, c, h, w)
        return self.loss(torch.log(ensemble_probs), mixture_dist) / (h * w)


class OracleLoss2d(nn.Module):
    def __init__(self, weight=None, reduce=True, size_average=True):
        super().__init__()
        weight = torch.Tensor(weight)

        self.loss = nn.NLLLoss(weight, reduce=reduce, size_average=size_average)

    def forward(self, outputs, targets):
        """
        :param outputs: Prediction of the model
        :param targets: ground-truth
        :return:
        """


        # producing the oracle mask to consider true predictions
        _outputs = outputs.max(1)[1]
        oracle_mask = (_outputs == targets).float()
        # outputs = _outputs * oracle_mask.type(FloatTensor)
        outputs = outputs * oracle_mask.unsqueeze(1)

        # return self.loss(_outputs, targets)
        return self.loss(F.log_softmax(outputs, 1), targets)


def get_citerion(lossname, **kwargs):
    if lossname == 'crossentropy':
        criterion = CrossEntropyLoss2d(**kwargs)
    elif lossname == 'oracle':
        criterion = OracleLoss2d(**kwargs)
    else:
        raise NotImplementedError
    return criterion


class JSDLoss(nn.Module):

    def __init__(self, reduce=True, size_average=False):
        super().__init__()

        self.loss = nn.KLDivLoss(reduce=reduce, size_average=size_average)

    def forward(self, ensemble_probs):
        # n: number of distributions
        # b: batch size
        # c: number of classes
        # (h, w): image dimensions
        n, b, c, h, w = ensemble_probs.shape
        mixture_dist = ensemble_probs.mean(0)
        entropy_mixture = torch.log(ensemble_probs).mean(0)


        # JSD = the entropy of the mixture - the mixture of the entropy
        # return self.loss(torch.log(ensemble_probs), mixture_dist) / (h * w)
        return self.loss(entropy_mixture, mixture_dist) / (h * w)


if __name__ == '__main__':
    img = torch.randn(1,3,44,44).float()
    gt = torch.randint(0,3,(1,44,44)).long()
    criterion = OracleLoss2d()
    loss = criterion(img,gt)


