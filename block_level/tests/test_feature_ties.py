from __future__ import annotations

import unittest
from array import array
from pathlib import Path

from block_level.config import FingerprintConfig
from block_level.fingerprint import FeatureSelector, FingerprintBuilder


def postings_for(candidates: dict[str, int]) -> dict[str, array[int]]:
    return {
        gram: array("I", (index for index in range(4) if mask & (1 << index)))
        for gram, mask in candidates.items()
    }


class FeatureTieTest(unittest.TestCase):
    def test_internal_entropy_prefers_frequency_then_name_at_each_step(self) -> None:
        # f wins by entropy. The remaining candidates have equal second-step
        # entropy; b and z occur in three units, while a occurs in one.
        candidates = {"f": 0b0011, "a": 0b0001, "z": 0b0111, "b": 0b1011}
        expected = ["f", "b"]
        self.assertEqual(
            FeatureSelector.internal_entropy(4, candidates.copy(), 2), expected
        )
        self.assertEqual(
            FeatureSelector.internal_entropy(
                4, candidates.copy(), 2, postings_for(candidates)
            ),
            expected,
        )

    def test_local_internal_entropy_uses_frequency_tie_break(self) -> None:
        config = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method="fingerprint_internal_entropy_equivalence_classes",
            subblock_size_rows=1,
            feature_selection_scope="local",
        )
        units = (
            frozenset({"f", "a", "z", "b"}),
            frozenset({"f", "z", "b"}),
            frozenset({"z"}),
            frozenset({"b"}),
        )
        selected = FeatureSelector(config).select_block_local_internal_entropy(
            {42: units}, 2
        )
        self.assertEqual(selected, {42: (("f",), ("b",))})

    def test_internal_entropy_first_step_tie(self) -> None:
        candidates = {"a": 0b0001, "z": 0b0111}
        for postings in (None, postings_for(candidates)):
            with self.subTest(native=postings is not None):
                self.assertEqual(
                    FeatureSelector.internal_entropy(
                        4, candidates.copy(), 1, postings
                    ),
                    ["z"],
                )

    def test_other_entropy_selectors_use_the_same_tie_order(self) -> None:
        candidates = {"f": 0b0011, "a": 0b0001, "z": 0b0111, "b": 0b1011}
        self.assertEqual(FeatureSelector.local_split(4, candidates.copy(), 2), ["f", "b"])
        self.assertEqual(
            FeatureSelector.distribution_entropy(4, candidates.copy(), 2),
            ["f", "b"],
        )
        self.assertEqual(
            FeatureSelector.within_block_joint_entropy((4,), candidates.copy(), 1),
            ["b"],
        )

    def test_equivalent_ngrams_choose_lexicographic_representative(self) -> None:
        representatives, aliases = FeatureSelector.collapse_equivalent(
            {"z": 0b0011, "a": 0b0011}
        )
        self.assertEqual(representatives, {"a": 0b0011})
        self.assertEqual(aliases, {"a": ("a", "z")})
    def test_feature_and_query_ngrams_share_reference_canonicalization(self) -> None:
        config = FingerprintConfig(
            widths=(3,),
            ngram_size=2,
            feature_selection_method="fingerprint_internal_entropy_equivalence_classes",
        )
        builder = FingerprintBuilder(config, 1, Path("metadata"), Path("results"))

        self.assertEqual(
            list(builder.iter_ngrams("CAFÉ")), ["ca", "af", "fe"]
        )
        self.assertEqual(
            builder.encode_query(
                "CAFÉ", {"ca": 0, "af": 1, "fe": 2}
            ),
            0b111,

        )


if __name__ == "__main__":
    unittest.main()
