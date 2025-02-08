
import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia.filters as kn_filters
import kornia.morphology as kn_morph
import torchvision.transforms.functional as TF



######################################################################################### Depth

########## SSI - MSE Loss ##########
'''def scale_and_shift_mae(matrix, mask):
    M = mask.sum()  
    assert M > 0  
    t = torch.median(matrix[mask])
    s = torch.sum(torch.abs(matrix[mask]- t)) / M
    # assert s > 0
    matrix_norm = (matrix[mask] - t) / (s + 1e-6)

    #print(f't:\t {t:.4f}, s:\t {s:.4f}')
    #print(f'matrix_norm.max:\t {matrix_norm.max():.4f}, matrix_norm.min:\t {matrix_norm.min():.4f}')
    return matrix_norm'''
    
def scale_and_shift_mae(pred, gt, mask):
    """
    Aligns predicted depth (relative) to GT depth using scale-and-shift optimization.
    """
    valid_pred = pred[mask]
    valid_gt = gt[mask]

    if valid_pred.numel() == 0 or valid_gt.numel() == 0:  # No valid pixels
        return pred  # Return unchanged

    # Solve for scale (s) and shift (t) using MAE optimization
    s = torch.sum(valid_gt * valid_pred) / torch.sum(valid_pred ** 2)
    t = torch.median(valid_gt - s * valid_pred)

    # Apply scale and shift
    pred_aligned = s * pred + t

    # Clip to GT range
    pred_aligned = torch.clamp(pred_aligned, min=0, max=80)

    return pred_aligned

def midas_mse_loss(pred_depth, gt_depth, mask, reduction):

    M = torch.sum(mask, (1, 2))
    res = pred_depth - gt_depth
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor
    

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 -
    # a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] -
                  a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] +
                  a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class MidasLoss(nn.Module):
    def __init__(self):
        super().__init__()        
        self.reduction = reduction_batch_based
    def forward(self, pdepth, gdepth, mask=None):
        mask = gdepth > 0 if mask is None else mask
        scale, shift = compute_scale_and_shift(pdepth, gdepth, mask)
        pred_ssi = scale.view(-1, 1, 1) * pdepth + shift.view(-1, 1, 1)
        loss = midas_mse_loss(pred_ssi, gdepth, mask, self.reduction)
        return loss
    


#####################HDN PR LOSS#####################
#num_bins in {1, 2, 4}
def normalize_depth(depth, context_indices):

    norm_depth = torch.zeros_like(depth)

    for _, indices in enumerate(context_indices):
        context_depth = depth[indices]
        median_depth = torch.median(context_depth)
        s_depth = torch.mean(torch.abs(context_depth - median_depth))
        norm_depth[indices] = (context_depth - median_depth) / (s_depth + 1e-6)
    
    return norm_depth

class HDNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_depth, gt_depth, num_bins, mask=None):
        if mask is None:
            mask = gt_depth > 0
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

        bins = torch.linspace(0, 1, num_bins + 1).to(pred_depth.device)
        context_indices = [torch.where((gt_depth >= bins[i]) & (gt_depth < bins[i + 1])) for i in range(num_bins)]
        
        pred_norm  = normalize_depth(pred_depth, context_indices)
        gt_norm = normalize_depth(gt_depth, context_indices)

        hdn_loss = 0
        for indices in context_indices:
            hdn_loss += torch.mean(torch.abs(pred_norm[indices] - gt_norm[indices]))
        #print(hdn_loss)
        hdn_loss /= num_bins

        return hdn_loss

######### MGS Loss #########
class ImageDerivative():
    def __init__(self, device=None):
        # seprable kernel: first derivative, second prefiltering
        tap_3 = torch.tensor([[0.425287, -0.0000, -0.425287], [0.229879, 0.540242, 0.229879]])
        tap_5 = torch.tensor([[0.109604,  0.276691,  0.000000, -0.276691, -0.109604], [0.037659,  0.249153,  0.426375,  0.249153,  0.037659]])
        tap_7 = torch.tensor([0])
        tap_9 = torch.tensor([[0.0032, 0.0350, 0.1190, 0.1458, -0.0000, -0.1458, -0.1190, -0.0350, -0.0032], [0.0009, 0.0151, 0.0890, 0.2349, 0.3201, 0.2349, 0.0890, 0.0151, 0.0009]])
        tap_11 = torch.tensor([0])
        tap_13 = torch.tensor([[0.0001, 0.0019, 0.0142, 0.0509, 0.0963, 0.0878, 0.0000, -0.0878, -0.0963, -0.0509, -0.0142, -0.0019, -0.0001],
                                [0.0000, 0.0007, 0.0071, 0.0374, 0.1126, 0.2119, 0.2605, 0.2119, 0.1126, 0.0374, 0.0071, 0.0007, 0.0000]])
        self.kernels=[tap_3, tap_5, tap_7, tap_9, tap_11, tap_13]

        # sending them to device
        if device is not None:
            self.to_device(device)

    def to_device(self,device):
        self.kernels = [kernel.to(device) for kernel in self.kernels]

    def __call__(self, img, t_id):
        # 
        # img : B*C*H*W
        # t_id : tap radius [for example t_id=1 will use the tap 3] 
        
        if t_id == 3 or t_id == 5:
            assert False, "Not Implemented"
        return self.forward(img, t_id)
    
    def forward(self, img, t_id=1):
        kernel = self.kernels[t_id-1]
        
        p = kernel[1:2,...]
        d1 = kernel[0:1,...]
        
        # B*C*H*W
        grad_x = kn_filters.filter2d_separable(img, p, d1, border_type='reflect', normalized=False, padding='same')
        grad_y = kn_filters.filter2d_separable(img, d1, p, border_type='reflect', normalized=False, padding='same')

        return (grad_x,grad_y)
    

    
class MSGLoss():
    def __init__(self,scales=4,taps=[2,2,1,1], k_size=[3,5,7,9], device=None):
        self.n_scale = scales
        self.taps = taps
        self.k_size = k_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        assert len(self.taps) == self.n_scale, 'number of scales and number of taps must be the same'
        assert len(self.k_size) == self.n_scale, 'number of scales and number of kernels must be the same'

        self.imgDerivative = ImageDerivative()

        self.erod_kernels = []
        for tap in self.taps:
            kernel = torch.ones(2*tap+1, 2*tap+1)
            self.erod_kernels.append(kernel)

        if self.device is not None:
            self.to_device(self.device)

    def to_device(self, device):
        self.imgDerivative.to_device(device)
        self.device = device
        self.erod_kernels = [kernel.to(device) for kernel in self.erod_kernels]
    
    def __call__(self, output, target, mask=None):
        return self.forward(output, target, mask)

    def forward(self, output, target, mask):

        if output.ndim == 3:
            output = output.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        
        diff = output - target
        
        if mask is None:
            mask = torch.ones(diff.shape[0],1, diff.shape[2], diff.shape[3])
            mask = mask.to(self.device)

        loss = 0
        for i in range(self.n_scale):
            # resize with antialiase
            mask_resized = torch.floor(self.resize_aa(mask.float(), i)+0.001)
            # erosion to mask out pixels that are effected by unkowns
            mask_resized = kn_morph.erosion(mask_resized, self.erod_kernels[i])
            
            diff_resized = self.resize_aa(diff, i)
            
            # compute grads
            grad_mag = self.gradient_mag(diff_resized, i)
            
            # mean over channels
            grad_mag = torch.mean(grad_mag, dim=1, keepdim=True) 

            # average the per pixel diffs
            loss += torch.sum(mask_resized * grad_mag) / (torch.sum(mask_resized) * grad_mag.shape[1])
        
        loss /= self.n_scale
        return loss
    
    def resize_aa(self,img, scale):
            if scale == 0:
                return img
            # blurred = TF.gaussian_blur(img, self.k_size[scale])
            # scaled = blurred[:, :, ::2**scale, ::2**scale]
            blurred = img
            scaled = torch.nn.functional.interpolate(blurred, scale_factor=1/(2**scale),mode='bilinear', align_corners=True, antialias=True)
            return scaled
    
    def gradient_mag(self, diff, scale):
        # B*C*H*W
        grad_x, grad_y = self.imgDerivative(diff, self.taps[scale])

        # B*C*H*W
        grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-7)

        return grad_magnitude
        

######################################################################################### Instance

def bbox_loss(pred, target, reduction='mean'):
    # pred : B, 4, H, W
    # target : B, 4, H, W

    bbox_loss = F.smooth_l1_loss(pred, target, reduction=reduction)
    return bbox_loss

def mask_loss(pred, target, reduction='mean'):
    # pred : B, 1, H, W
    # target : B, 1, H, W

    mask_loss = F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
    return mask_loss

def class_loss(pred, target, reduction='mean'):
    # pred : B, C, H, W
    # target : B, H, W

    class_loss = F.cross_entropy(pred, target, reduction=reduction)
    return class_loss


class InstanceSegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        bbox = bbox_loss(pred['bbox'], target['bbox'])
        mask = mask_loss(pred['mask'], target['mask'])
        class_ = class_loss(pred['class'], target['class'])

        alpha1, alpha2, alpha3 = 1, 1, 1

        loss = alpha1 * bbox + alpha2 * mask + alpha3 * class_
        return loss

