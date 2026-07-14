# Source Snapshot

This runtime bundle was copied from the independently developed model iteration:

```text
/Users/yuanx/Documents/00_developer/crystal_ai/model_iterations/v2.0.0
```

The copied inference source is limited to:

```text
src/library.py
src/network.py
src/predict.py
```

The copied runtime assets are limited to the checkpoint, vocabularies, constrained-decoding
metadata, reactant-element mapping, legal reactant lists, and normalization mappings.
Training JSONL files, raw data, validation samples, plots, and training reports are excluded.

The checkpoint copied into this bundle has SHA-256:

```text
8c230a91bfe13e5f725fe44538a1c008ffc8626b06675058d5de5eaeab3d2e3e
```

Future integration code must load the bundle through `PredictionService`, verify this digest
against `MANIFEST.json`, and must not refer to the source path above at runtime.
