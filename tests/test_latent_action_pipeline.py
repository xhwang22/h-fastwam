import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from fastwam.models.hfastwam.hfastwam import HFastWAM
from fastwam.models.hfastwam.latent_action_decoder import LatentActionDecoder
from fastwam.trainer import Wan22Trainer


class _TrainabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(2, 2)
        self.latent_action_decoder = nn.Linear(2, 2)
        self.proprio_encoder = None
        self.use_visual_encoder = False
        self.freeze_language_expert = False
        self.freeze_video_expert = False
        self.freeze_action_expert = False


class LatentActionPipelineTest(unittest.TestCase):
    def test_fixed_contract_validation_and_direct_mode(self):
        direct, decoder = HFastWAM._validate_latent_action_configs(
            None, None, {"action_dim": 14}, 14, 1664
        )
        self.assertEqual(direct, {})
        self.assertEqual(decoder, {})

        config = {
            **HFastWAM.LATENT_ACTION_CONTRACT,
            "enabled": True,
            "oracle_probabilities": [1, 0.75, 0.5, 0.25, 0],
            "decoder_loss_type": "smooth_l1",
            "decoder_loss_beta": 0.5,
        }
        decoder_config = {
            "latent_dim": 32,
            "proprio_dim": 14,
            "visual_dim": 1664,
            "num_latents": 8,
            "substeps_per_latent": 4,
            "action_dim": 14,
        }
        validated, validated_decoder = HFastWAM._validate_latent_action_configs(
            config, decoder_config, {"action_dim": 32}, 14, 1664
        )
        self.assertTrue(validated["enabled"])
        self.assertEqual(validated_decoder, decoder_config)

        with self.assertRaisesRegex(ValueError, "latent_horizon=8"):
            HFastWAM._validate_latent_action_configs(
                {**config, "latent_horizon": 7},
                decoder_config,
                {"action_dim": 32},
                14,
                1664,
            )

    def test_dictconfig_contract_validation(self):
        config = OmegaConf.create(
            {
                **HFastWAM.LATENT_ACTION_CONTRACT,
                "enabled": True,
                "oracle_probabilities": [1, 0.75, 0.5, 0.25, 0],
                "decoder_loss_type": "smooth_l1",
                "decoder_loss_beta": 1.0,
            }
        )
        decoder = OmegaConf.create(
            {
                "latent_dim": 32,
                "proprio_dim": 14,
                "visual_dim": 1664,
                "num_latents": 8,
                "substeps_per_latent": 4,
                "action_dim": 14,
            }
        )
        validated, validated_decoder = HFastWAM._validate_latent_action_configs(
            config, decoder, {"action_dim": 32}, 14, 1664
        )
        self.assertIsInstance(validated, dict)
        self.assertIsInstance(validated_decoder, dict)
        self.assertTrue(validated["enabled"])

    def test_clean_latent_math(self):
        noisy = torch.tensor([[[3.0, 5.0]]])
        velocity = torch.tensor([[[2.0, 4.0]]])
        clean = HFastWAM._estimate_clean_latent(
            noisy, velocity, torch.tensor([250.0]), 1000
        )
        torch.testing.assert_close(clean, torch.tensor([[[2.5, 4.0]]]))

    def test_oracle_schedule_and_deterministic_selection(self):
        model = HFastWAM.__new__(HFastWAM)
        nn.Module.__init__(model)
        model.latent_action_decoder = nn.Identity()
        model.latent_action_enabled = True
        model.latent_action_config = {
            "oracle_probabilities": [1.0, 0.75, 0.5, 0.25, 0.0]
        }
        for epoch, expected in enumerate((1.0, 0.75, 0.5, 0.25, 0.0, 0.0)):
            model.set_training_epoch(epoch)
            self.assertEqual(model._latent_action_oracle_probability(), expected)

        oracle = torch.ones(16, 8, 32)
        generated = torch.zeros_like(oracle)
        first, first_mask = model._select_decoder_latent(
            oracle,
            generated,
            0.5,
            generator=torch.Generator().manual_seed(7),
        )
        second, second_mask = model._select_decoder_latent(
            oracle,
            generated,
            0.5,
            generator=torch.Generator().manual_seed(7),
        )
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(first_mask, second_mask))

    def test_masked_smooth_l1_uses_physical_padding(self):
        prediction = torch.zeros(1, 2, 2)
        target = torch.tensor([[[1.0, 1.0], [100.0, 100.0]]])
        loss = HFastWAM._compute_latent_action_decoder_loss(
            prediction,
            target,
            torch.tensor([[False, True]]),
            beta=1.0,
        )
        torch.testing.assert_close(loss, torch.tensor(0.5))

    def test_checkpoint_metadata(self):
        model = HFastWAM.__new__(HFastWAM)
        nn.Module.__init__(model)
        model.latent_action_enabled = True
        model.latent_action_config = {
            **HFastWAM.LATENT_ACTION_CONTRACT,
            "latent_cache_signature": "cache-v1",
        }
        model.latent_action_decoder_config = {"d_model": 32}
        metadata = model._checkpoint_metadata()
        self.assertEqual(metadata["checkpoint_schema_version"], 2)
        self.assertEqual(metadata["action_representation"], "latent")
        self.assertEqual(metadata["latent_cache_signature"], "cache-v1")
        self.assertEqual(metadata["physical_action_horizon"], 32)

    def test_manifest_provenance_updates_checkpoint_metadata(self):
        model = HFastWAM.__new__(HFastWAM)
        nn.Module.__init__(model)
        model.latent_action_enabled = True
        model.latent_action_config = {**HFastWAM.LATENT_ACTION_CONTRACT}
        model.latent_action_decoder_config = {"d_model": 32}
        model.set_latent_action_cache_manifest(
            {
                "signature": "a" * 64,
                "signature_payload": {
                    "normalization": {
                        "type": "standardize",
                        "mean": [0.0] * 32,
                        "std": [1.0] * 32,
                    },
                    "dreamdojo": {
                        "git_revision": "c" * 40,
                        "checkpoint_revision": "d" * 40,
                        "checkpoint_sha256": "b" * 64,
                    },
                },
            }
        )
        metadata = model._checkpoint_metadata()
        self.assertEqual(metadata["latent_cache_signature"], "a" * 64)
        self.assertEqual(metadata["dreamdojo_code_revision"], "c" * 40)
        self.assertEqual(metadata["dreamdojo_checkpoint_revision"], "d" * 40)
        self.assertEqual(metadata["dreamdojo_checkpoint_hash"], "b" * 64)
        self.assertEqual(metadata["latent_normalization_stats"]["type"], "standardize")

    def test_manifest_provenance_rejects_config_mismatch(self):
        model = HFastWAM.__new__(HFastWAM)
        nn.Module.__init__(model)
        model.latent_action_enabled = True
        model.latent_action_config = {
            **HFastWAM.LATENT_ACTION_CONTRACT,
            "dreamdojo_code_revision": "expected",
        }
        model.latent_action_decoder_config = {"d_model": 32}
        with self.assertRaisesRegex(ValueError, "dreamdojo_code_revision"):
            model.set_latent_action_cache_manifest(
                {
                    "signature": "a" * 64,
                    "signature_payload": {
                        "normalization": {
                            "type": "standardize",
                            "mean": [0.0] * 32,
                            "std": [1.0] * 32,
                        },
                        "dreamdojo": {
                            "git_revision": "actual",
                            "checkpoint_revision": "d" * 40,
                            "checkpoint_sha256": "b" * 64,
                        },
                    },
                }
            )

    def test_checkpoint_round_trip_restores_decoder(self):
        def make_model():
            model = HFastWAM.__new__(HFastWAM)
            nn.Module.__init__(model)
            model.language_expert = nn.Linear(2, 2)
            model.mot = nn.Linear(2, 2)
            model.latent_action_decoder = nn.Linear(2, 2)
            model.latent_action_enabled = True
            model.latent_action_config = {
                **HFastWAM.LATENT_ACTION_CONTRACT,
                "latent_cache_signature": "cache-v1",
            }
            model.latent_action_decoder_config = {"d_model": 2}
            model.proprio_encoder = None
            model.use_visual_encoder = False
            model.fixed_target_encoder_enabled = False
            model.__dict__["_fixed_teacher_handle"] = None
            model._training_phase = "full"
            model.torch_dtype = torch.float32
            return model

        source = make_model()
        with torch.no_grad():
            source.latent_action_decoder.weight.fill_(3.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            source.save_checkpoint(str(checkpoint))
            target = make_model()
            target.load_checkpoint(str(checkpoint), strict=True)
        torch.testing.assert_close(
            target.latent_action_decoder.weight,
            source.latent_action_decoder.weight,
        )

    def test_rejects_unsupported_checkpoint_schema(self):
        model = HFastWAM.__new__(HFastWAM)
        nn.Module.__init__(model)
        model.latent_action_enabled = False
        model.language_expert = nn.Linear(2, 2)
        model.mot = nn.Linear(2, 2)
        model.proprio_encoder = None
        model.use_visual_encoder = False
        model.fixed_target_encoder_enabled = False
        model.__dict__["_fixed_teacher_handle"] = None
        model._training_phase = "full"
        model.torch_dtype = torch.float32
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model.save_checkpoint(str(checkpoint))
            payload = torch.load(checkpoint, weights_only=True)
            payload["checkpoint_metadata"]["checkpoint_schema_version"] = 999
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint schema"):
                model.load_checkpoint(str(checkpoint), strict=True)

    def test_legacy_direct_checkpoint_still_loads_strictly(self):
        model = HFastWAM.__new__(HFastWAM)
        nn.Module.__init__(model)
        model.latent_action_enabled = False
        model.language_expert = nn.Linear(2, 2)
        model.mot = nn.Linear(2, 2)
        model.proprio_encoder = None
        model.use_visual_encoder = False
        model._training_phase = "full"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            torch.save(
                {
                    "language_expert": model.language_expert.state_dict(),
                    "mot": model.mot.state_dict(),
                    "training_phase": "full",
                },
                checkpoint,
            )
            model.load_checkpoint(str(checkpoint), strict=True)

    def test_decoder_trainability_component(self):
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.trainable_components = ["dit", "latent_action_decoder"]
        trainer.freeze_visual_encoder = True
        model = _TrainabilityModel()
        trainer._apply_dit_only_train_mode(model)
        self.assertTrue(all(parameter.requires_grad for parameter in model.dit.parameters()))
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.latent_action_decoder.parameters())
        )

    def test_curriculum_uses_decoder_train_mode(self):
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.trainable_components = ["dit", "latent_action_decoder"]
        trainer.freeze_visual_encoder = True
        model = HFastWAM.__new__(HFastWAM)
        nn.Module.__init__(model)
        model.dit = nn.Linear(2, 2)
        model.latent_action_decoder = nn.Linear(2, 2)
        model.proprio_encoder = None
        model.use_visual_encoder = False
        model.freeze_language_expert = False
        model.freeze_video_expert = False
        model.freeze_action_expert = False
        model.latent_action_enabled = True
        model.latent_action_config = {
            "oracle_probabilities": [1.0, 0.75, 0.5, 0.25, 0.0]
        }
        model._training_epoch = 1

        trainer._apply_dit_only_train_mode(model)
        self.assertFalse(model.training)
        self.assertTrue(model.latent_action_decoder.training)
        self.assertEqual(model._latent_action_oracle_probability(), 0.75)

        model.eval()
        self.assertEqual(model._latent_action_oracle_probability(), 0.0)

    def test_small_decoder_preserves_external_shape(self):
        decoder = LatentActionDecoder(
            latent_dim=32,
            proprio_dim=14,
            visual_dim=16,
            num_latents=8,
            substeps_per_latent=4,
            action_dim=14,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        output = decoder(
            torch.randn(2, 8, 32),
            torch.randn(2, 14),
            torch.randn(2, 16, 1, 1, 1),
            flatten_output=True,
        )
        self.assertEqual(output.shape, (2, 32, 14))


if __name__ == "__main__":
    unittest.main()
