import copy
import tempfile
import unittest
from pathlib import Path

import torch

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.utils.latent_action_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    LATENT_ACTION_IS_PAD_SHAPE,
    LATENT_ACTION_SHAPE,
    LatentActionCacheError,
    atomic_write_json,
    canonical_signature,
    load_latent_action,
    load_latent_action_cache_manifest,
    validate_latent_action_cache_manifest,
    validate_latent_action_tensors,
    write_latent_action_shard,
)


class LatentActionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temporary_directory.name)
        self.normalization = {
            "type": "standardize",
            "mean": torch.arange(32, dtype=torch.float32).tolist(),
            "std": torch.full((32,), 2.0).tolist(),
        }
        self.signature_payload = {
            "cache_format": CACHE_FORMAT,
            "cache_version": CACHE_VERSION,
            "dataset": {"name": "fixture", "length": 3},
            "normalization": self.normalization,
        }
        self.signature = canonical_signature(self.signature_payload)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_complete_cache(self):
        latent = torch.arange(3 * 8 * 32, dtype=torch.float32).reshape(3, 8, 32)
        padding = torch.zeros(3, 8, dtype=torch.bool)
        padding[1, 3] = True
        records = [
            write_latent_action_shard(
                self.cache_dir,
                0,
                [0, 1],
                latent[:2],
                padding[:2],
            ),
            write_latent_action_shard(
                self.cache_dir,
                1,
                [2],
                latent[2:],
                padding[2:],
            ),
        ]
        mean = torch.tensor(self.normalization["mean"])
        std = torch.tensor(self.normalization["std"])
        manifest = {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "complete": True,
            "signature": self.signature,
            "signature_payload": self.signature_payload,
            "dataset_length": 3,
            "shard_size": 2,
            "num_shards": 2,
            "latent_action_shape": list(LATENT_ACTION_SHAPE),
            "latent_action_is_pad_shape": list(LATENT_ACTION_IS_PAD_SHAPE),
            "normalization": copy.deepcopy(self.normalization),
            "shards": records,
        }
        atomic_write_json(self.cache_dir / "manifest.json", manifest)
        return latent, padding, manifest, mean, std

    def test_canonical_signature_is_order_independent(self):
        left = {"outer": {"b": 2, "a": 1}, "value": [3, 4]}
        right = {"value": [3, 4], "outer": {"a": 1, "b": 2}}
        self.assertEqual(canonical_signature(left), canonical_signature(right))

    def test_manifest_validates_signature_payload_version_and_expected_signature(self):
        _, _, manifest, _, _ = self._write_complete_cache()
        validated = load_latent_action_cache_manifest(
            self.cache_dir,
            expected_length=3,
            expected_signature=self.signature,
        )
        self.assertEqual(validated["signature"], self.signature)

        bad_version = copy.deepcopy(manifest)
        bad_version["version"] = CACHE_VERSION + 1
        with self.assertRaisesRegex(LatentActionCacheError, "version mismatch"):
            validate_latent_action_cache_manifest(bad_version)

        tampered_payload = copy.deepcopy(manifest)
        tampered_payload["signature_payload"]["dataset"]["length"] = 4
        with self.assertRaisesRegex(LatentActionCacheError, "does not match"):
            validate_latent_action_cache_manifest(tampered_payload)

        with self.assertRaisesRegex(LatentActionCacheError, "signature mismatch"):
            validate_latent_action_cache_manifest(
                manifest,
                expected_signature="0" * 64,
            )

    def test_manifest_rejects_unsigned_normalization_tampering(self):
        _, _, manifest, _, _ = self._write_complete_cache()
        tampered = copy.deepcopy(manifest)
        tampered["normalization"]["mean"][0] = 123.0
        with self.assertRaisesRegex(
            LatentActionCacheError,
            "must exactly match `signature_payload.normalization`",
        ):
            validate_latent_action_cache_manifest(tampered)

        missing_signed_normalization = copy.deepcopy(manifest)
        missing_signed_normalization["signature_payload"].pop("normalization")
        missing_signed_normalization["signature"] = canonical_signature(
            missing_signed_normalization["signature_payload"]
        )
        with self.assertRaisesRegex(
            LatentActionCacheError,
            "must exactly match `signature_payload.normalization`",
        ):
            validate_latent_action_cache_manifest(missing_signed_normalization)

    def test_manifest_rejects_tampered_shard_hash(self):
        self._write_complete_cache()
        shard = self.cache_dir / "shard_00000000.safetensors"
        with shard.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(LatentActionCacheError, "SHA256 mismatch"):
            load_latent_action_cache_manifest(self.cache_dir)

    def test_sharded_round_trip_reads_by_global_sample_index(self):
        latent, padding, _, mean, std = self._write_complete_cache()
        manifest = load_latent_action_cache_manifest(self.cache_dir)

        for index in range(3):
            actual, actual_padding = load_latent_action(
                self.cache_dir,
                manifest,
                index,
            )
            expected = (latent[index] - mean) / std
            expected[padding[index]] = 0.0
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_padding, padding[index])
            self.assertTrue(actual.is_contiguous())

        with self.assertRaises(IndexError):
            load_latent_action(self.cache_dir, manifest, 3)

    def test_writer_rejects_duplicate_and_missing_indices(self):
        latent = torch.zeros(2, 8, 32)
        padding = torch.zeros(2, 8, dtype=torch.bool)
        for indices in ([4, 4], [4, 6]):
            with self.subTest(indices=indices), self.assertRaisesRegex(
                LatentActionCacheError,
                "contiguous and increasing",
            ):
                write_latent_action_shard(
                    self.cache_dir,
                    0,
                    indices,
                    latent,
                    padding,
                )

    def test_reader_rejects_wrong_stored_sample_index(self):
        _, _, manifest, _, _ = self._write_complete_cache()
        bad_record = write_latent_action_shard(
            self.cache_dir,
            0,
            [1, 2],
            torch.zeros(2, 8, 32),
            torch.zeros(2, 8, dtype=torch.bool),
        )
        bad_record["index_start"] = 0
        bad_record["index_stop"] = 2
        manifest["shards"][0] = bad_record
        with self.assertRaisesRegex(LatentActionCacheError, "stores sample 1, expected 0"):
            load_latent_action(self.cache_dir, manifest, 0)

    def test_tensor_shape_dtype_and_finiteness_validation(self):
        valid_latent = torch.zeros(2, 8, 32, dtype=torch.float16)
        valid_padding = torch.zeros(2, 8, dtype=torch.bool)
        actual_latent, actual_padding = validate_latent_action_tensors(
            valid_latent.transpose(1, 2).transpose(1, 2),
            valid_padding,
            batch=True,
        )
        self.assertTrue(actual_latent.is_contiguous())
        self.assertTrue(actual_padding.is_contiguous())

        invalid_cases = (
            (torch.zeros(2, 7, 32), valid_padding, "must end"),
            (torch.zeros(2, 8, 32, dtype=torch.int64), valid_padding, "floating point"),
            (valid_latent, torch.zeros(2, 8), "must be bool"),
            (valid_latent, torch.zeros(1, 8, dtype=torch.bool), "batch size mismatch"),
        )
        for latent, padding, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                LatentActionCacheError,
                message,
            ):
                validate_latent_action_tensors(latent, padding, batch=True)

        nonfinite = valid_latent.clone()
        nonfinite[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(LatentActionCacheError, "non-finite"):
            validate_latent_action_tensors(nonfinite, valid_padding, batch=True)

    def test_dataset_padding_mask_composes_adjacent_images_and_action_groups(self):
        image_padding = torch.tensor(
            [False, False, True, False, False, False, False, False, True]
        )
        action_padding = torch.zeros(32, dtype=torch.bool)
        action_padding[20] = True

        actual = RobotVideoDataset._expected_latent_action_is_pad(
            image_padding,
            action_padding,
        )
        expected = torch.tensor(
            [False, True, True, False, False, True, False, True]
        )
        torch.testing.assert_close(actual, expected)

        with self.assertRaisesRegex(LatentActionCacheError, "9 image"):
            RobotVideoDataset._expected_latent_action_is_pad(
                torch.zeros(8, dtype=torch.bool),
                action_padding,
            )
        with self.assertRaisesRegex(LatentActionCacheError, "32 physical"):
            RobotVideoDataset._expected_latent_action_is_pad(
                image_padding,
                torch.zeros(31, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
