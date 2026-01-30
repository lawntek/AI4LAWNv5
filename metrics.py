#metrics
import numpy as np
import torch

def metrics_seg(pred, target):
    
    num_classes = 3
    assert pred.shape == target.shape

    pred = torch.flatten(pred)
    target = torch.flatten(target)

    mIoU=0
    ioU= [0]*num_classes
    for i in range(num_classes):
        intersection = torch.sum((pred == i) & (target == i))
        union = torch.sum((pred == i) | (target == i))
        if union == 0:
            iou = 0
        else:
            iou = intersection.float() / union.float()

        ioU[i] = iou
        mIoU += iou

    mIoU = mIoU / num_classes
    precision = torch.sum(pred == target).float() / len(target)
    
    return mIoU, precision, ioU


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        
        num_classes = 3
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)
        target = torch.nn.functional.one_hot(target, num_classes).float()

        pred = torch.nn.functional.softmax(pred, dim=1)
        pt = torch.sum(pred * target, dim=1)
        loss = - (1 - pt) ** self.gamma * torch.log(pt)

        return torch.mean(loss)
    
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
    
        num_classes = 3
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)
        #target = torch.nn.functional.one_hot(target, num_classes).float()

        pred = torch.nn.functional.softmax(pred, dim=1)
        smooth = 1
        intersection = torch.sum(pred * target, dim=0)
        union = torch.sum(pred, dim=0) + torch.sum(target, dim=0)
        dice = (2 * intersection + smooth) / (union + smooth)

        return 1 - torch.mean(dice)

