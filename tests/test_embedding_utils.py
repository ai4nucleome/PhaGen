"""CPU pooling checks with independent coordinate and valid-count oracles."""

import unittest

import torch

from phagen.embedding_utils import pool_hidden_states

STAGE_LENGTHS = (128, 64, 16)
STAGE_DIMS = (512, 256, 196)
CONTENT_VALUES = (1.0, 4.0, 7.0)
START_VALUES = (10.0, 15.0, 20.0)

# Counts include learned start tokens; start counts are listed separately.
# These are independent expectations at nucleotide, 16-base and 1024-base edges.
COUNTS = {
    1: ((2, 2, 2), (1, 1, 1)),
    16: ((2, 2, 17), (1, 1, 1)),
    17: ((2, 3, 19), (1, 1, 2)),
    65: ((2, 6, 70), (1, 1, 5)),
    1024: ((2, 65, 1088), (1, 1, 64)),
    1025: ((3, 67, 1090), (1, 2, 65)),
}


def make_hidden_states(length, dtype=torch.float32, padding_value=9000.0):
    """Assign tokens by their genomic coordinates, without constructing masks."""
    patches = (length + 1023) // 1024
    coarse = torch.full((1, patches + 1, 512), padding_value, dtype=dtype)
    middle = torch.full((1, patches, 65, 256), padding_value, dtype=dtype)
    fine = torch.full((1, patches, 64, 17, 196), padding_value, dtype=dtype)
    coarse[0, 0] = START_VALUES[0]
    for patch in range(patches):
        coarse[0, patch + 1] = CONTENT_VALUES[0]
        middle[0, patch, 0] = START_VALUES[1]
        for group in range(64):
            first_base = patch * 1024 + group * 16
            # A padding-only group's learned start token has a distinct sentinel.
            fine[0, patch, group, 0] = -padding_value
            if first_base < length:
                middle[0, patch, group + 1] = CONTENT_VALUES[1]
                fine[0, patch, group, 0] = START_VALUES[2]
            for base in range(16):
                if first_base + base < length:
                    fine[0, patch, group, base + 1] = CONTENT_VALUES[2]
    return [coarse, middle, fine]


class EmbeddingPoolingTests(unittest.TestCase):
    def test_valid_start_tokens_and_partial_patch_boundaries(self):
        for length, (valid_counts, start_counts) in COUNTS.items():
            with self.subTest(length=length):
                hidden = make_hidden_states(length)
                actual = pool_hidden_states(hidden, length, STAGE_LENGTHS)
                self.assertEqual(actual.shape, (1, 964))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, "cpu")
                chunks = torch.split(actual, STAGE_DIMS, dim=-1)
                for stage, (chunk, count, starts) in enumerate(
                    zip(chunks, valid_counts, start_counts)
                ):
                    expected = (
                        (count - starts) * CONTENT_VALUES[stage]
                        + starts * START_VALUES[stage]
                    ) / count
                    torch.testing.assert_close(
                        chunk, torch.full_like(chunk, expected), rtol=1e-6, atol=1e-6
                    )

    def test_padding_and_padding_only_start_values_do_not_affect_result(self):
        for length in (65, 1025):
            with self.subTest(length=length):
                small = pool_hidden_states(
                    make_hidden_states(length, padding_value=100.0), length, STAGE_LENGTHS
                )
                large = pool_hidden_states(
                    make_hidden_states(length, padding_value=1e12), length, STAGE_LENGTHS
                )
                torch.testing.assert_close(small, large, rtol=0, atol=0)

    def test_half_precision_accumulation_remains_finite_float32(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                # The fine-stage sum exceeds the finite float16 range even
                # though the correct mean remains representable.
                hidden = make_hidden_states(1025, dtype=dtype)
                for stage in hidden:
                    stage.fill_(60000.0)
                actual = pool_hidden_states(hidden, 1025, STAGE_LENGTHS)
                expected_value = torch.tensor(60000.0, dtype=dtype).float().item()
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(
                    actual, torch.full((1, 964), expected_value), rtol=1e-6, atol=1e-3
                )

    def test_invalid_lengths_and_stage_shapes_are_rejected(self):
        hidden = make_hidden_states(65)
        for length in (0, -1, True, 1.5, 131073):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    pool_hidden_states(hidden, length, STAGE_LENGTHS)
        with self.assertRaises(ValueError):
            pool_hidden_states(hidden[:2], 65, STAGE_LENGTHS)
        with self.assertRaises(ValueError):
            pool_hidden_states(hidden, 65, (128, 64, 0))
        bad_shape = [hidden[0], hidden[1], hidden[2][..., :-1, :]]
        with self.assertRaises(ValueError):
            pool_hidden_states(bad_shape, 65, STAGE_LENGTHS)


if __name__ == "__main__":
    unittest.main()
