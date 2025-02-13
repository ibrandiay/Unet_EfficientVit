import torch


def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        IoU = IoU + (1-IoU1)

    return IoU/b


class Iou(torch.nn.Module):
    def __init__(self, size_average = True):
        super(Iou, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


def iou_loss(pred, label):
    iou_loss_ = Iou(size_average=True)
    iou_out = iou_loss_(pred, label)
    print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out
