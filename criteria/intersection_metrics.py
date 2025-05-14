import torch

def metrics_torch(y_true, y_pred, metric_name,
                  metric_type='standard', drop_last=True, mean_per_class=False, verbose=False):
    """
    Compute mean metrics of two segmentation masks, via PyTorch.

    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)

    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot
          encoding y_pred.
          The 'naive' version returns mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation).
        mean_per_class: return mean along batch axis for each class.
        verbose: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.

    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    assert y_true.shape == y_pred.shape, 'Input masks should be the same shape, instead are {}, {}'.format(y_true.shape, y_pred.shape)
    assert len(y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)

    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')

    num_classes = y_pred.shape[1]
    drop_last = drop_last and num_classes > 1

    if not flag_soft:
        if num_classes > 1:
            y_pred = torch.stack([(torch.argmax(y_pred, dim=1)==i) for i in range(num_classes)]).permute(1,2,3,0)
            y_true = torch.stack([(torch.argmax(y_true, dim=1)==i) for i in range(num_classes)]).permute(1,2,3,0)
            # y_pred = torch.nn.functional.one_hot(y_pred, num_classes=num_classes).permute(0, 3, 1, 2).float()
            # y_true = torch.nn.functional.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            y_pred = (y_pred > 0).int()
            y_true = (y_true > 0).int()

    # Intersection and union shapes are (batch_size, n_classes)
    axes = (1, 2)  # W,H axes of each image
    intersection = torch.sum(torch.abs(y_pred * y_true), dim=axes)  # or, y_pred.logical_and(y_true)
    mask_sum = torch.sum(torch.abs(y_true), dim=axes) + torch.sum(torch.abs(y_pred), dim=axes)
    union = mask_sum - intersection  # or, y_pred.logical_or(y_true)

    if verbose:
        print('intersection (pred*true), intersection (pred&true), union (pred+true-inters), union (pred|true)')
        print(intersection, torch.sum(y_pred.logical_and(y_true), dim=axes), union, torch.sum(y_pred.logical_or(y_true), dim=axes))

    smooth = 0.001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # Define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask = (union != 0).int()

    if drop_last:
        metric = metric[:, :-1]
        mask = mask[:, :-1]

    # Return mean metrics: remaining axes are (batch, classes)
    if mean_per_class:
        if flag_naive_mean:
            return metric  # Return all values
        else:
            # Mean only over non-absent classes in batch
            return (torch.sum(metric * mask, dim=0) + smooth) / (torch.sum(mask, dim=0) + smooth)
    else:
        if flag_naive_mean:
            return torch.mean(metric)
        else:
            # Mean only over non-absent classes
            class_count = torch.sum(mask, dim=0)
            return torch.mean(torch.sum(metric * mask, dim=0)[class_count != 0] / (class_count[class_count != 0]))