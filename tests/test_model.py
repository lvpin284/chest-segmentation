"""Tests for the U-Net segmentation model."""
import pytest
import torch
from src.models.unet import UNet, SegmentationLoss


class TestUNet:
    """Unit tests for the U-Net architecture."""

    def test_output_shape_matches_input(self):
        """Model output spatial dimensions must equal the input spatial dimensions."""
        model = UNet(in_channels=1, num_classes=1)
        x = torch.randn(1, 1, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256), f"Unexpected shape: {out.shape}"

    def test_grayscale_and_rgb_inputs(self):
        """Model should accept both 1-channel and 3-channel inputs."""
        for c in (1, 3):
            model = UNet(in_channels=c, num_classes=1)
            x = torch.randn(2, c, 128, 128)
            out = model(x)
            assert out.shape == (2, 1, 128, 128)

    def test_odd_spatial_dimensions(self):
        """Model must handle odd height/width gracefully (padding logic)."""
        model = UNet(in_channels=1, num_classes=1)
        x = torch.randn(1, 1, 255, 255)
        out = model(x)
        assert out.shape == (1, 1, 255, 255)

    def test_predict_mask_binary_values(self):
        """predict_mask must return only 0.0 and 1.0 values."""
        model = UNet(in_channels=1, num_classes=1)
        x = torch.randn(1, 1, 64, 64)
        mask = model.predict_mask(x)
        unique_vals = torch.unique(mask).tolist()
        assert set(unique_vals).issubset({0.0, 1.0}), f"Non-binary values: {unique_vals}"

    def test_predict_mask_shape(self):
        """predict_mask output shape must equal input spatial shape."""
        model = UNet(in_channels=1, num_classes=1)
        x = torch.randn(2, 1, 128, 128)
        mask = model.predict_mask(x)
        assert mask.shape == (2, 1, 128, 128)

    def test_base_features_scaling(self):
        """Model should work with a smaller base_features (e.g. 16) for memory efficiency."""
        model = UNet(in_channels=1, num_classes=1, base_features=16)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)


class TestSegmentationLoss:
    """Unit tests for the BCE + Dice combined loss."""

    def test_loss_is_scalar(self):
        """Loss output must be a scalar tensor."""
        loss_fn = SegmentationLoss()
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"

    def test_perfect_prediction_low_loss(self):
        """When predictions exactly match targets, loss should be very small."""
        loss_fn = SegmentationLoss()
        # All-one target; use very large positive logit to simulate confidence
        targets = torch.ones(2, 1, 32, 32)
        logits = torch.ones(2, 1, 32, 32) * 20.0
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05, f"Loss too high for perfect prediction: {loss.item()}"

    def test_loss_gradients(self):
        """Loss must be differentiable (gradients flow back to logits)."""
        loss_fn = SegmentationLoss()
        logits = torch.randn(1, 1, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (1, 1, 32, 32)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None, "No gradients computed"
