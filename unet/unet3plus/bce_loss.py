import torch.nn as nn


def bce_loss(pred, label):
	bce_loss_ = nn.BCELoss(size_average=True)
	bce_out = bce_loss_(pred, label)
	print("bce_loss:", bce_out.data.cpu().numpy())
	return bce_out
