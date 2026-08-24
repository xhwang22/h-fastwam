import json

import torch
import torch.nn as nn
from safetensors.torch import save_file

from fastwam.models.dreamdojo_lam import encode_dreamdojo_latent_actions
from fastwam.models.hfastwam.hfastwam import HFastWAM
from fastwam.models.hfastwam.hfastwam_latent_action import HFastWAMLatentAction
from fastwam.models.hfastwam.language_expert import LanguageExpert
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.latent_action_dit import LatentActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.visual_encoder import BaseVisualEncoder
from fastwam.utils.latent_action_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    latent_action_shard_path,
    latent_action_tensor_key,
    load_latent_action,
    load_latent_action_cache_manifest,
)


def _tiny_latent_action_dit() -> LatentActionDiT:
    return LatentActionDiT(
        hidden_dim=32,
        latent_dim=4,
        latent_horizon=8,
        ffn_dim=64,
        context_dim=16,
        context_spatial_pool=2,
        freq_dim=16,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=8,
        num_layers=2,
    )


def test_latent_action_pre_dit_shapes_and_bidirectional_mask():
    model = _tiny_latent_action_dit()
    latent_actions = torch.randn(2, 8, 4)
    timestep = torch.rand(2)
    visual_context = torch.randn(2, 16, 1, 6, 8)

    state = model.pre_dit(
        latent_tokens=latent_actions,
        timestep=timestep,
        context_latents=visual_context,
    )

    assert state["tokens"].shape == (2, 8, 32)
    assert state["context"].shape == (2, 12, 32)
    assert state["context_mask"].shape == (2, 8, 12)
    assert state["meta"]["tokens_per_frame"] == 8
    assert model.build_video_to_video_mask(8, 8, latent_actions.device).all()


def test_hfastwam_action_rows_read_all_latent_action_tokens():
    owner = type("MaskOwner", (), {"video_expert": _tiny_latent_action_dit()})()
    mask = HFastWAM._build_full_attention_mask(
        owner,
        task_len=3,
        subtask_len=0,
        video_seq_len=8,
        action_seq_len=6,
        video_tokens_per_frame=8,
        device=torch.device("cpu"),
    )

    latent_rows = mask[3:11]
    action_rows = mask[11:17]
    assert latent_rows[:, 3:11].all()
    assert not latent_rows[:, 11:17].any()
    assert action_rows[:, 3:11].all()
    assert action_rows[:, 11:17].all()


def test_latent_action_cache_loads_and_normalizes(tmp_path):
    latent_action = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "complete": True,
        "dataset_length": 1,
        "shard_size": 8,
        "latent_horizon": 3,
        "latent_dim": 4,
        "mean": [2.0, 3.0, 4.0, 5.0],
        "std": [2.0, 2.0, 2.0, 2.0],
    }
    with (tmp_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    save_file(
        {latent_action_tensor_key(0): latent_action},
        str(latent_action_shard_path(tmp_path, 0)),
    )

    loaded_manifest = load_latent_action_cache_manifest(
        tmp_path,
        expected_length=1,
        expected_horizon=3,
        expected_dim=4,
    )
    loaded = load_latent_action(
        tmp_path,
        loaded_manifest,
        0,
        normalize=True,
    )
    expected = (
        latent_action - torch.tensor(manifest["mean"])
    ) / torch.tensor(manifest["std"])
    torch.testing.assert_close(loaded, expected)


class _FakeDreamDojo(nn.Module):
    def encode(self, pairs):
        delta = pairs[:, 1].mean(dim=(1, 2, 3))
        delta = delta - pairs[:, 0].mean(dim=(1, 2, 3))
        return {"z_rep": delta[:, None, None, None].expand(-1, 1, 1, 32)}


def test_online_dreamdojo_encoding_preserves_transition_order():
    frame_values = torch.linspace(-1.0, 1.0, 33)
    video = frame_values[None, None, :, None, None].expand(2, 3, 33, 8, 8)
    latent_actions = encode_dreamdojo_latent_actions(
        _FakeDreamDojo(),
        video,
        pair_batch_size=13,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )
    preprocessed_latent_actions = encode_dreamdojo_latent_actions(
        _FakeDreamDojo(),
        video,
        pair_batch_size=13,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        preprocess_all_frames=True,
    )

    assert latent_actions.shape == (2, 32, 32)
    expected_delta = ((frame_values[1:] - frame_values[:-1]) * 0.5)
    torch.testing.assert_close(latent_actions[0, :, 0], expected_delta)
    torch.testing.assert_close(latent_actions[1], latent_actions[0])
    torch.testing.assert_close(
        preprocessed_latent_actions,
        latent_actions,
    )


def test_checkpoint_provenance_accepts_legacy_v1_target_contract():
    provenance = {
        "checkpoint_sha256": "checkpoint",
        "target_contract": "dreamdojo_adjacent_pair_z_mu_32d_v1",
    }
    owner = type(
        "ProvenanceOwner",
        (),
        {"dreamdojo_provenance": provenance},
    )()
    metadata = {
        "video_target_representation": "dreamdojo_latent_action",
        "dreamdojo_target": {"checkpoint_sha256": "checkpoint"},
    }

    HFastWAMLatentAction._validate_checkpoint_metadata(
        owner,
        metadata,
        strict=True,
        path="legacy.pt",
    )


class _FakeVisualEncoder(BaseVisualEncoder):
    def __init__(self):
        super().__init__()
        self.z_dim = 8
        self.upsampling_factor = 2
        self.temporal_downsample_factor = 1
        self.projection = nn.Identity()
        self.backbone = nn.Identity()
        self._freeze_backbone = True

    def encode(
        self,
        videos,
        device="cpu",
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        return_pre_standardise=False,
    ):
        del tiled, tile_size, tile_stride
        first = videos[:, :, :1].to(device=device)
        pooled = torch.nn.functional.adaptive_avg_pool3d(first, (1, 4, 4))
        latents = pooled.mean(dim=1, keepdim=True).expand(-1, 8, -1, -1, -1)
        if return_pre_standardise:
            return latents, latents
        return latents


def test_action_loss_backpropagates_through_latent_action_stream():
    latent_expert = LatentActionDiT(
        hidden_dim=32,
        latent_dim=4,
        latent_horizon=8,
        ffn_dim=64,
        context_dim=8,
        context_spatial_pool=2,
        freq_dim=16,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=8,
        num_layers=2,
    )
    action_expert = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=8,
        num_layers=2,
    )
    language_expert = LanguageExpert(
        hidden_dim=32,
        num_heads=4,
        attn_head_dim=8,
        ffn_dim=64,
        num_layers=2,
        vocab_size=64,
        max_task_len=8,
        max_subtask_len=8,
        eps=1.0e-6,
        dtype=torch.float32,
    )
    mot = MoT(
        mixtures={
            "language": language_expert,
            "video": latent_expert,
            "action": action_expert,
        },
        mot_checkpoint_mixed_attn=False,
        strict_expert_compat=True,
    )
    model = HFastWAMLatentAction(
        language_expert=language_expert,
        video_expert=latent_expert,
        action_expert=action_expert,
        mot=mot,
        vae=None,
        text_dim=16,
        proprio_dim=5,
        device="cpu",
        torch_dtype=torch.float32,
        loss_lambda_language=0.0,
        loss_lambda_video=0.0,
        loss_lambda_action=1.0,
        knowledge_insulation=False,
        action_loss_detach_video_expert=False,
        freeze_language_expert=True,
        visual_encoder=_FakeVisualEncoder(),
    )
    flat_sample = {
        "video": torch.randn(2, 3, 2, 8, 8),
        "latent_actions": torch.randn(2, 8, 4),
        "action": torch.randn(2, 8, 3),
        "proprio": torch.randn(2, 8, 5),
        "task_token_ids": torch.randint(0, 64, (2, 3)),
        "subtask_token_ids": torch.empty(2, 0, dtype=torch.long),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }
    sample = {
        "segments": {
            **{
                key: value.unsqueeze(1)
                for key, value in flat_sample.items()
            },
            "segment_mask": torch.ones(2, 1, dtype=torch.bool),
        }
    }

    loss, loss_dict = model.training_loss(sample)
    loss.backward()

    assert loss_dict["loss_latent_action"] == 0.0
    assert loss_dict["loss_action"] > 0.0
    assert latent_expert.action_encoder.weight.grad is not None
    assert latent_expert.action_encoder.weight.grad.abs().sum() > 0
