"""
FFT Loss Functions

This module implements various FFT (Fast Fourier Transform) based loss functions
for image restoration tasks. These losses operate in the frequency domain to
capture structural information that spatial losses might miss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    This loss computes the L1 distance between the FFT of predicted and target images.
    It helps capture high-frequency details and structural information.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                           f'Supported ones are: none | mean | sum')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # Compute 2D FFT for each channel
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        # Stack real and imaginary parts
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        # Compute L1 loss
        if weight is not None:
            loss = torch.mean(torch.abs(pred_fft - target_fft) * weight)
        else:
            loss = torch.mean(torch.abs(pred_fft - target_fft))

        # Apply reduction
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass  # Already mean-reduced in the computation above

        return self.loss_weight * loss


class FFTL1Loss(nn.Module):
    """Pure FFT L1 loss without complex representation.

    This is a simpler implementation that computes the L1 distance between
    the magnitudes of FFT coefficients.

    Args:
        loss_weight (float): Loss weight for FFT L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction mode: {reduction}. "
                           f"Supported modes are: 'none', 'mean', 'sum'")
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        """Forward pass."""
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        diff = torch.abs(pred_fft - target_fft)

        # Apply the reduction
        if self.reduction == 'mean':
            loss = diff.mean()
        elif self.reduction == 'sum':
            loss = diff.sum()
        else:  # 'none'
            loss = diff

        return self.loss_weight * loss


class StableFFTLoss(nn.Module):
    """
    A stable L1 loss in the frequency domain, combined with a spatial L1 loss.
    This implementation includes stability improvements like ignoring DC component.

    Args:
        loss_weight (float): Loss weight for the combined loss. Default: 1.0.
        alpha (float): Weight for the frequency component. Default: 0.01.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        norm (str): Normalization method for FFT. Default: 'ortho'.
        ignore_dc (bool): Whether to ignore the DC component. Default: True.
    """

    def __init__(self, loss_weight=1.0, alpha=0.01, reduction='mean', norm='ortho', ignore_dc=True):
        super(StableFFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # Weight for the frequency component
        self.reduction = reduction
        self.norm = norm
        self.ignore_dc = ignore_dc

    def forward(self, pred, target):
        """Forward pass."""
        # Spatial L1 Loss
        spatial_loss = F.l1_loss(pred, target, reduction=self.reduction)

        # FFT of prediction and target
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm=self.norm)
        target_fft = torch.fft.fft2(target, dim=(-2, -1), norm=self.norm)

        # Separate real and imaginary parts
        pred_fft_real = pred_fft.real
        pred_fft_imag = pred_fft.imag
        target_fft_real = target_fft.real
        target_fft_imag = target_fft.imag

        # Stability improvements
        if self.ignore_dc:
            pred_fft_real[..., 0, 0] = target_fft_real[..., 0, 0]
            pred_fft_imag[..., 0, 0] = target_fft_imag[..., 0, 0]

        # L1 loss for real and imaginary parts
        freq_loss_real = F.l1_loss(pred_fft_real, target_fft_real, reduction=self.reduction)
        freq_loss_imag = F.l1_loss(pred_fft_imag, target_fft_imag, reduction=self.reduction)

        freq_loss = freq_loss_real + freq_loss_imag

        # Final combined loss
        combined_loss = spatial_loss + self.alpha * freq_loss

        return self.loss_weight * combined_loss


class FreqLoss(nn.Module):
    """Combined frequency and spatial L1 loss.

    This loss combines a frequency domain L1 loss with a spatial L1 loss
    to capture both high-frequency details and overall structure.

    Args:
        loss_weight (float): Loss weight for the combined loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        """Forward pass."""
        # FFT-based loss
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        fft_loss = torch.mean(torch.abs(diff))

        # Spatial L1 loss
        spatial_loss = self.l1_loss(pred, target)

        # Combined loss with small weight for FFT component
        loss = fft_loss * 0.01 + spatial_loss

        return self.loss_weight * loss


class FreqNormLoss(nn.Module):
    """Frequency loss with ortho normalization.

    Similar to FreqLoss but uses ortho normalization for more stable training.

    Args:
        loss_weight (float): Loss weight for the combined loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqNormLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        """Forward pass."""
        # FFT-based loss with ortho normalization
        diff = torch.fft.rfft2(pred, norm='ortho') - torch.fft.rfft2(target, norm='ortho')
        fft_loss = torch.mean(torch.abs(diff))

        # Spatial L1 loss
        spatial_loss = self.l1_loss(pred, target)

        # Combined loss
        loss = fft_loss * 0.01 + spatial_loss

        return self.loss_weight * loss