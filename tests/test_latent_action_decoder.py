import inspect
import unittest

import torch

from fastwam.models.hfastwam.latent_action_decoder import LatentActionDecoder


class LatentActionDecoderTest(unittest.TestCase):
    @staticmethod
    def _decoder(**kwargs):
        config = {
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
        }
        config.update(kwargs)
        return LatentActionDecoder(**config)

    def test_default_architecture_configuration(self):
        parameters = inspect.signature(LatentActionDecoder).parameters
        expected_defaults = {
            "latent_dim": 32,
            "proprio_dim": 14,
            "visual_dim": 1664,
            "num_latents": 8,
            "substeps_per_latent": 4,
            "action_dim": 14,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 2048,
            "dropout": 0.1,
        }
        for name, expected in expected_defaults.items():
            self.assertEqual(parameters[name].default, expected)

    def test_structured_and_flattened_output_shapes(self):
        torch.manual_seed(0)
        decoder = self._decoder().eval()
        latent = torch.randn(2, 8, 32)
        proprio = torch.randn(2, 14)
        visual = torch.randn(2, 1664, 1, 2, 3)

        structured = decoder(latent, proprio, visual)
        flattened = decoder(latent, proprio, visual, flatten_output=True)

        self.assertEqual(structured.shape, (2, 8, 4, 14))
        self.assertEqual(flattened.shape, (2, 32, 14))
        torch.testing.assert_close(structured.reshape(2, 32, 14), flattened)

    def test_gradients_reach_all_inputs_and_decoder_parameters(self):
        torch.manual_seed(1)
        decoder = self._decoder()
        latent = torch.randn(2, 8, 32, requires_grad=True)
        proprio = torch.randn(2, 14, requires_grad=True)
        visual = torch.randn(2, 1664, 1, 2, 2, requires_grad=True)

        decoder(latent, proprio, visual).square().mean().backward()

        for value in (latent, proprio, visual):
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(decoder.substep_queries.grad)
        self.assertGreater(decoder.substep_queries.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(decoder.visual_projection.weight.grad)

    def test_supported_visual_layouts(self):
        torch.manual_seed(2)
        decoder = self._decoder().eval()
        latent = torch.randn(1, 8, 32)
        proprio = torch.randn(1, 14)
        canonical = torch.randn(1, 1664, 1, 2, 3)
        channel_first_map = canonical[:, :, 0]
        token_last = channel_first_map.permute(0, 2, 3, 1).reshape(1, 6, 1664)
        channel_first_tokens = token_last.transpose(1, 2)

        expected = decoder(latent, proprio, canonical)
        for visual in (channel_first_map, token_last, channel_first_tokens):
            torch.testing.assert_close(decoder(latent, proprio, visual), expected)

        pooled_output = decoder(latent, proprio, torch.randn(1, 1664))
        self.assertEqual(pooled_output.shape, (1, 8, 4, 14))

    def test_padded_latents_do_not_affect_valid_outputs(self):
        torch.manual_seed(3)
        decoder = self._decoder().eval()
        latent = torch.randn(2, 8, 32)
        changed = latent.clone()
        changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 100
        latent_is_pad = torch.tensor(
            [[False, False, False, False, False, False, True, True]] * 2
        )
        proprio = torch.randn(2, 14)
        visual = torch.randn(2, 1664)

        expected = decoder(
            latent,
            proprio,
            visual,
            latent_is_pad=latent_is_pad,
            flatten_output=True,
        )
        actual = decoder(
            changed,
            proprio,
            visual,
            latent_is_pad=latent_is_pad,
            flatten_output=True,
        )

        torch.testing.assert_close(expected[:, :24], actual[:, :24])
        self.assertTrue(torch.equal(actual[:, 24:], torch.zeros_like(actual[:, 24:])))

    def test_input_validation(self):
        decoder = self._decoder()
        latent = torch.randn(2, 8, 32)
        proprio = torch.randn(2, 14)
        visual = torch.randn(2, 1664, 1, 2, 2)

        invalid_cases = (
            ("latent", (torch.randn(2, 7, 32), proprio, visual), "latent.*shape"),
            ("proprio", (latent, torch.randn(2, 13), visual), "current_proprio.*shape"),
            ("batch", (latent, proprio, torch.randn(1, 1664, 1, 2, 2)), "batch size"),
            ("future visual", (latent, proprio, torch.randn(2, 1664, 2, 2, 2)), "time|1, H, W"),
            ("visual channels", (latent, proprio, torch.randn(2, 10, 3)), "visual_state.*must"),
        )
        for name, args, pattern in invalid_cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, pattern):
                decoder(*args)

        with self.assertRaisesRegex(TypeError, "floating-point"):
            decoder(latent.to(torch.int64), proprio, visual)
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._decoder(d_model=30, nhead=8)

    def test_rejects_ambiguous_flattened_visual_layout(self):
        decoder = self._decoder(visual_dim=4)
        with self.assertRaisesRegex(ValueError, "unambiguously"):
            decoder(
                torch.randn(1, 8, 32),
                torch.randn(1, 14),
                torch.randn(1, 4, 4),
            )


if __name__ == "__main__":
    unittest.main()
