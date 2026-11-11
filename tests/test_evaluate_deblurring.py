import unittest

import numpy as np
import torch

from scripts.evaluate_deblurring import predict_deblurgan, predict_nafnet


class CaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.inputs = inputs.detach().clone()
        return inputs


class PredictionNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.linspace(0, 1, 8 * 8 * 3, dtype=np.float32)
        self.image = values.reshape(8, 8, 3)
        self.device = torch.device("cpu")

    def test_nafnet_receives_training_range(self) -> None:
        model = CaptureModel()

        restored = predict_nafnet(model, self.image, self.device)

        captured = model.inputs
        self.assertIsNotNone(captured)
        assert captured is not None
        np.testing.assert_allclose(
            captured[0].permute(1, 2, 0).numpy(), self.image
        )
        np.testing.assert_allclose(restored, self.image)

    def test_deblurgan_receives_training_range_and_maps_output_back(self) -> None:
        model = CaptureModel()

        restored = predict_deblurgan(model, self.image, self.device)

        expected_input = torch.from_numpy(self.image.transpose(2, 0, 1)).mul(2).sub(1)
        captured = model.inputs
        self.assertIsNotNone(captured)
        assert captured is not None
        torch.testing.assert_close(captured[0], expected_input)
        np.testing.assert_allclose(restored, self.image, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
