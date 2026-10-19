"""Boundary and input-contract checks without a model or network access."""

import unittest

from phagen.sequence_utils import extract_contigs


class TokenizerStub:
    bos_token_id = 0
    eos_token_id = 5
    mask_token_id = 6
    pad_token_id = 7
    unk_token_id = 8
    ids = {"A": 1, "T": 2, "C": 3, "G": 4}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def decode(self, *args, **kwargs):
        raise AssertionError("Extraction must use base token IDs, not decode or strip spaces.")


class TensorStub:
    def __init__(self, values, ndim=1):
        self.values = values
        self.ndim = ndim

    def tolist(self):
        return self.values


class ExtractContigsTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerStub()

    def test_returns_complete_records_in_order(self):
        ids = [4, 5, 0, 1, 3, 4, 2, 5, 0, 2, 1, 5]
        self.assertEqual(extract_contigs(ids, self.tokenizer), ["ACGT", "TA"])
        self.assertEqual(ids, [4, 5, 0, 1, 3, 4, 2, 5, 0, 2, 1, 5])

    def test_nested_bos_restarts_and_trailing_span_is_dropped(self):
        ids = [0, 1, 1, 0, 3, 4, 5, 0, 2, 1]
        self.assertEqual(extract_contigs(ids, self.tokenizer), ["CG"])

    def test_non_dna_content_discards_whole_span_then_recovers(self):
        for invalid_id in (6, 7, 8, 9, -1):
            with self.subTest(invalid_id=invalid_id):
                ids = [0, 1, invalid_id, 2, 5, 0, 3, 4, 5]
                self.assertEqual(extract_contigs(ids, self.tokenizer), ["CG"])

    def test_length_counts_bases_and_is_inclusive(self):
        ids = [0, 5, 0, 1, 5, 0, 1, 2, 5, 0, 1, 2, 3, 5]
        self.assertEqual(extract_contigs(ids, self.tokenizer), ["A", "AT", "ATC"])
        self.assertEqual(extract_contigs(ids, self.tokenizer, min_length=2), ["AT", "ATC"])
        self.assertEqual(extract_contigs(ids, self.tokenizer, min_length=4), [])

    def test_tensor_protocol_and_tuple(self):
        ids = [0, 1, 3, 4, 2, 5]
        self.assertEqual(extract_contigs(TensorStub(ids), self.tokenizer), ["ACGT"])
        self.assertEqual(extract_contigs(tuple(ids), self.tokenizer), ["ACGT"])
        self.assertEqual(extract_contigs([], self.tokenizer), [])

    def test_invalid_length_limits_raise_value_error(self):
        for minimum in (None, True, False, 0, -1, 1.5, "2"):
            with self.subTest(minimum=minimum):
                with self.assertRaises(ValueError):
                    extract_contigs([0, 1, 5], self.tokenizer, min_length=minimum)

    def test_invalid_shapes_and_token_types_raise_value_error(self):
        invalid_inputs = (
            None, 1, "015", b"015", {0, 1, 5}, {"ids": [0, 1, 5]},
            [[0, 1, 5]], [0, 1.0, 5], [0, True, 5], [0, "1", 5],
            TensorStub([[0, 1, 5]], ndim=2), TensorStub(1, ndim=0),
            TensorStub([[0, 1, 5]]),
        )
        for ids in invalid_inputs:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    extract_contigs(ids, self.tokenizer)

    def test_invalid_tokenizer_boundary_or_base_ids_raise_value_error(self):
        for tokenizer in (
            object(),
            type("SameBoundary", (TokenizerStub,), {"eos_token_id": 0})(),
            type("MissingBase", (TokenizerStub,), {"ids": dict.fromkeys("ACGT", 8)})(),
            type("UnknownBase", (TokenizerStub,), {"ids": {"A": 1, "C": 3, "G": 4, "T": 8}})(),
            type("UnmappedBase", (TokenizerStub,), {"ids": {"A": 1}})(),
            type("BoundaryBase", (TokenizerStub,), {"ids": {"A": 0, "C": 1, "G": 2, "T": 3}})(),
        ):
            with self.subTest(tokenizer=tokenizer):
                with self.assertRaises(ValueError):
                    extract_contigs([0, 1, 5], tokenizer)


if __name__ == "__main__":
    unittest.main()
