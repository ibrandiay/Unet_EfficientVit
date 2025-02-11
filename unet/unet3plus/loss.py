"""
All the loss functions used to train the model are defined here

@author: Ibra Ndiaye
@date: 17/10/2023

"""


import torch
from typing import List


def roll_window(tensor, dim = 2, size = 25, step = None) -> torch.Tensor:
    """uses tensor.unfold to return a tensor containing sliding windows over the source.
    windows don't overlap by default, use the step parameter to change it."""
    if step is None :
        step = size
    result = tensor.unfold(dim,size,step)
    result = result.unfold(dim+1,size,step)
    return result


def batch_im_cov(I1,I2) -> torch.float:
    assert I1.nelement()==I2.nelement(), "tensors need to have the same size"

    iflat, i2flat = I1.flatten(start_dim=1).float(), I2.flatten(start_dim=1).float()
    meanI = torch.mean(iflat,dim=1)
    meanI2 = torch.mean(i2flat,dim=1)

    return torch.mean((iflat-meanI.unsqueeze(-1))*(i2flat - meanI2.unsqueeze(-1)),dim=1)


def ssim(image_1, image_2, beta=1, gamma=1, **rollwin_kwargs) -> torch.float:
    """computes mean single-scale Structural SIMilarity index between two image-like tensors (NCHW) using a sliding
    window."""

    C1 = 1e-4
    C2 = 9e-4

    windows_1 = roll_window(image_1, **rollwin_kwargs)
    windows_2 = roll_window(image_2, **rollwin_kwargs)

    total_ssim = torch.zeros(image_1.shape[0], device=image_1.get_device())

    for i in range(windows_1.shape[2]):
        for j in range(windows_1.shape[3]):

            m1, m2 = torch.mean(windows_1[..., i, j], dim=(1, 2, 3)), torch.mean(windows_2[..., i, j],dim=(1, 2, 3))
            s1, s2 = torch.std(windows_1[..., i, j], dim=(1, 2, 3)), torch.std(windows_2[..., i, j], dim=(1, 2, 3))

            C = (2*m1*m2 + C1) / (m1*m1 + m2*m2 + C1)
            S = (2*batch_im_cov(windows_1[..., i, j], windows_2[..., i, j]) + C2) / (s1*s1+s2*s2+C2)

            window_sim = C**beta * S**gamma
            if not torch.any(window_sim.isnan()):
                total_ssim += window_sim

    return total_ssim/ (windows_1.shape[2]*windows_2.shape[3]+0.0001)


def iou_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.float:
    """
    Calculates Intersection over Union. basic loss
    Args:
        pred: predictions
        targ: targets

    Returns:
        loss
    """
    pred_flat, targ_flat = pred.flatten(start_dim=1), targ.flatten(start_dim=1)
    intersection = torch.sum(pred_flat*targ_flat)
    union = pred_flat.sum()+targ_flat.sum() - intersection
    return 1 - (intersection+0.1)/(union+0.1)


def ms_ssim_loss(pred_scales, target, betas: List[int] = [1], gammas: List[int] = [1]) -> torch.float:
    """
	    Multi-scale similarity loss computed from the product of losses at similar scales and a final target image.
        Typically used to compare the intermediary outputs of the decoder branch and the ground truth mask.

	    Args:
            pred_scales: list of predictions
            target: ground truth
            betas: list of beta values
            gammas: list of gamma values

	    Returns:
		loss: the loss computed
	"""
    
    ssim_product= 1
    for i in range(len(pred_scales)):
        pred = pred_scales[i][:, 1:, ...]
        _beta = betas[min(i, len(betas)-1)]
        _gamma = gammas[min(i, len(gammas)-1)]
        ssim_product *= ssim(pred, target[:, 1:, ...], _beta, _gamma)

    return 1-ssim_product


