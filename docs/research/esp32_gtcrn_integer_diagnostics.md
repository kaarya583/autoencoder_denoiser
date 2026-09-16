# Training-only GTCRN quantization localization

`esp32_denoiser.gtcrn_integer_diagnostics` diagnoses a frozen float GTCRN
checkpoint without changing its weights or the calibrated integer graph.
It uses the same checkpoint-recipe audit as the existing probes: fixed
augmentation implementation hashes, training and validation manifest hashes,
source-group/ID/path/hash separation, selected asset hashes, crop hashes,
and the checkpoint's actual clean-identity probability. The development
manifest is read to check disjointness; its audio is never loaded.

The dataset is iterated exactly once to select 32 deterministic training
crops (default seed 483). Those frozen tensors are reused for normal min/max
calibration and a diagnostic pass over the first four crops. Diagnostics
inspect the first 64 causal frames of each crop, resetting every utterance.
Both the float shadow and integer graph receive identical complex64 network
spectra, including the checkpoint's optional frame-RMS normalization. The
crop count and frame limit are explicit CLI controls. No values are fitted
to development clips, and the tool reports no waveform quality score.

The report separates four comparisons:

- **Propagated edge error:** decoded integer activation versus the original
  float graph on the same spectrum. This includes earlier layers' errors.
- **Float edge grid loss:** the original float activation rounded directly
  to its assigned encoding. This isolates the resolution and range of that
  boundary before upstream integer errors.
- **Local operator error:** integer output versus the original float
  operator fed the actual decoded integer inputs and prior INT8 state.
  This excludes upstream error already present in those inputs. A frequency
  GRU call includes its internal recurrence across 33 bins.
- **Local error after reference output rounding:** the same local float
  reference rounded to the actual output encoding. Remaining differences
  reflect quantized weights, internal integer arithmetic, or intermediate
  recurrent/normalization boundaries. It is a localization clue, not a
  causal attribution of a waveform SI-SDR change.

All 188 graph boundaries and 84 operator calls are covered. Parameter
reports cover the 65 parameter-bearing operators, including per-output
folded affine weights and biases, PReLU slopes, both GRU directions and
LayerNorm gamma/beta. The tool decodes sigmoid codes as `(q+128)/255`, with
explicit zero/one endpoints, and recurrent/tanh outputs on their Q7 grid.
Rail contact is reported separately from float-reference values outside a
grid; touching a valid endpoint does not prove clipping. Relative errors
must be read alongside reference RMS and absolute error, especially for
quiet activations.

Example invocation:

```sh
python -m esp32_denoiser.gtcrn_integer_diagnostics \
  --checkpoint /path/to/frozen_gtcrn.pt \
  --manifest /path/to/development_manifest.jsonl \
  --calibration-crops 32 --seed 483 --probe-crops 4 \
  --max-frames-per-crop 64 --threads 1 \
  --output /path/to/training_integer_diagnostics.json
```

Six focused tests verify source immutability, all boundaries and error
decompositions, explicit probability/state decoding, detection of a known
weight corruption, and exception-safe restoration of instrumentation. The
real checkpoint fixture selects each crop exactly once and succeeds even
after development audio is deliberately made unreadable; final-test
manifests are rejected. This is a diagnostic tool, with no QAT recovery,
operator-class float bypass, physical MCU timing or denoising improvement
claimed.
