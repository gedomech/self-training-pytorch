import torch


def naiveway(predict_logit):
    return predict_logit.max(1)[1]
