# Recovered dataset export records

These small records were recovered from the separate `VCR_Lab` workspace on
2026-10-10. The raw TFRecord datasets, photos, archives, and the notebook with
hardcoded Roboflow credentials are not part of this repository.

- `test_augment_v2/`: Roboflow version-2 export records for the candidate
  MobileNet pipeline dataset, plus the associated Kaggle upload metadata.
- `nutrition_labels_no_augs_v2/`: Roboflow version-2 export records for the
  candidate external dataset.
- [dataset-manifest.json](dataset-manifest.json): metadata-copy hashes and
  local TFRecord split sizes, hashes, and framing counts.

The candidate training export contains 2,886 train, 120 validation, and 121
test records. The candidate external export contains 300 train, 53 validation,
and 45 test records. Their test counts match the saved evaluation JSONs, but
those JSONs do not include dataset hashes or executed commands. Count matches
alone do not prove historical dataset identity, overlap, or independence.
The framing scan did not execute training code, decode TensorFlow examples,
validate CRCs, or run an evaluation.

The Roboflow export notices declare CC BY 4.0. The Kaggle metadata for
`test_augment-2` instead lists CC0; this inconsistency is retained rather than
silently resolved. See [third-party attribution](../../THIRD_PARTY_NOTICES.md)
and [experimental provenance](../EXPERIMENTS.md) for the distinction between
recorded export terms and unverified source-image lineage.
