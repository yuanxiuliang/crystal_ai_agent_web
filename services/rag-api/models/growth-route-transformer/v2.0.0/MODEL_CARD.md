# Growth Route Transformer v2.0.0

## Purpose

`growth-route-transformer` generates up to three candidate single-crystal growth routes
from one chemical formula. It is an evidence source for the RAG workbench, not a source of
literature citations or a guarantee of experimental success.

## Runtime Bundle

The bundle is self-contained for inference:

```text
src/                         inference source snapshot
models/v2.0.0_run1.best.pth checkpoint
features/                    input and output vocabularies
lib/config/                  token and bin metadata
lib/rawLib/                  constrained-decoding reactant assets
data/mappings/               formula/reactant normalization mappings
```

The bundle intentionally excludes training data, raw records, validation samples, and
training reports. The checkpoint digest and evaluation summary are in `MANIFEST.json`.

## Model Contract

Input is a parseable chemical formula. The model returns constrained beam-search candidates
for `Flux` or `CVT` routes containing raw reactants, additives, binned temperature fields,
and an optional binned duration.

Temperatures and durations are discretized during training. A displayed bin midpoint is a
presentation value only; the evidence contract must preserve the original bin range.

The displayed Top-3 scores are normalized ranking weights within the returned candidates.
They are not calibrated confidence or experimental success probabilities.

## Known Limits

1. The model accepts a formula only. It does not condition on user equipment limits,
   pressure, atmosphere, or desired method.
2. It supports only `Flux` and `CVT` outputs.
3. Element-coverage constrained decoding does not prove chemical feasibility, safety, or
   thermodynamic stability.
4. Unknown formula tokens and out-of-domain formulas require explicit warnings.
5. Generated routes require literature comparison and experimental validation.

## Evaluation Snapshot

The formula-grouped held-out validation set has 542 samples. Top-1 method accuracy is 0.9207,
raw-reactant set F1 is 0.7990, additive set F1 is 0.6231, route exact match is 0.4391, and
Top-3 route hit is 0.5480. Temperature MAE is 129.2 C and duration MAE is 141.7 h.
