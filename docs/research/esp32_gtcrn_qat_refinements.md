# GTCRN range calibration and GPU QAT priorities

The next implementation should prioritize the normalized GTCRN candidate,
improve fixed activation grids using training audio, then train the complete
integer graph with matching forward boundaries. Keep the audited GRU,
LayerNorm, probability and C operator contracts. The evidence currently
points more strongly to activation resolution and propagated error than to
weight precision or a recurrent arithmetic defect.

## What the frozen evidence supports

The raw epoch-25 and normalized epoch-15 checkpoints are independently trained
models. On the same four development clips, the raw model changes from
10.4231 to 6.9450 dB SI-SDR improvement under complete NumPy INT8 inference;
the normalized model changes from 11.3961 to 10.5621 dB. These are small-cohort
observations, not a controlled isolation of normalization's causal effect or
a final quality result. Their files are under
`output/esp32/gtcrn_probes_01/{gtcrn_probe_inputs,gtcrn_normalized_probe_inputs}`.

The subsequent **training-only** diagnostics cover the first 64 frames of
four audited training crops (256 frames total), with identical spectra in
the float shadow and integer reference. The results are under
`output/esp32/gtcrn_probes_03/*/training_diagnostics.json`.

| Training diagnostic | Raw checkpoint | Normalized checkpoint |
|---|---:|---:|
| Input ERB activation exponent | 0 | −7 |
| Input signal RMS | 1.74363 | 0.037806 |
| Input quantization RMSE | 0.20534 | 0.002098 |
| Input SQNR | 18.58 dB | 25.11 dB |
| Zero input codes / 99,072 values | 79,357 | 43,912 |
| Largest folded matrix relative L2 error | 1.037% | 0.989% |

In the normalized model, encoder block 4's depthwise local output has signal
RMS 1.07257 and error RMS 0.07255. Rounding the float operator's output alone
accounts for RMS 0.07191; these are substantial signals, not a near-zero
relative-error artifact. Decoder block 1's attention energy has RMS 2.12726
and error RMS 0.14348, with **zero discrepancy after rounding the ideal local
reference onto the output grid**. Several PReLU and attention-product errors
show the same boundary-resolution pattern. The final mask's own direct Q7
rounding error is only about 0.00163 RMS, while propagated mask error is
0.19550 RMS, so changing its fixed endpoint encoding is not the first target.

Raw decoder block 1's energy has 16 out-of-range local reference values after
upstream integer distortion. A grid calibrated solely on original float
activations may miss that changed distribution. This is a reason to measure
the complete selected integer model after range selection, not to infer
quality from lower local MSE. Neither the four-clip development statistics
nor these training diagnostics are used to fit development data.

## Implemented: exact training-stream MSE grids

`esp32_denoiser/gtcrn_mse_calibration.py` is an isolated alternative to the
unchanged min/max calibration. It selects the same 32 checkpoint-recipe
training crops once (seed 483), retains those waveform tensors, and performs
two numerical float passes: existing min/max calibration, then candidate
SSE scoring over **every observed activation**. No activation histogram or
sampling approximation is used.

For each ordinary edge, the default candidates are its min/max exponent plus
`[-3,-2,-1,0,+1]`, restricted to supported grids and arithmetic bounds. The
original exponent is always included and wins exact ties. The selected
exponent minimizes local reconstruction SSE. GRU and grouped recurrent
outputs and final tanh stay Q7; sigmoid probabilities retain `(q+128)/255`.
GRU input limits, affine bias bounds and LayerNorm input/output constraints
are checked before packing. Jointly invalid LayerNorm choices are resolved
only among valid candidate pairs, with an explicit audit record. The final
whole graph is validated again.

Every edge records candidate MSE, clipping count/fraction, zero-code count,
maximum error, represented range and a shape-framed hash of its exact float32
activation stream. The audit also retains min/max grids, selected grids,
rejected candidates, source code hash, checkpoint/source hashes and the
audited crop/asset records. All grids are fixed for a deployed stream.

```sh
python -m esp32_denoiser.gtcrn_mse_calibration \
  --checkpoint /path/to/frozen_gtcrn.pt \
  --manifest /path/to/development_manifest.jsonl \
  --output /path/to/gtcrn_mse.bin \
  --calibration-crops 32 --calibration-seed 483 --threads 1
```

The command writes the real binary plus `.calibration.json` and size metadata
through the existing serializer. The binary binds the complete audit by
SHA256. Ten focused tests pass; 67 combined calibration, diagnostic, packed
export and native-graph tests pass. Tests include a known rare-clipping MSE
tradeoff, exact min/max replay, fixed contracts, determinism, source
immutability and matching NumPy/packed/C-neural waveform outputs. A real
checkpoint fixture proves one dataset iteration and no development-audio
reads by making those files undecodable. Final-test manifests are rejected.

Local SSE is not a guarantee of better speech quality. Adjacent edge grids,
skip paths and recurrent distributions interact, and a finer downstream
grid cannot reconstruct information already discarded upstream. Compare the
min/max and MSE **actual integer models** on the frozen primary, external and
clean development suites before promoting either. Preserve the untouched
final-test protocol.

## Next: reparameterization, then full QAT

If grid selection leaves large errors concentrated in channels with unequal
ranges, test positive channel equalization inside simple
`Conv → PReLU → Conv` chains. Multiplying the first layer's output and bias by
a positive diagonal scale D, and dividing the next convolution's input
weights by D, preserves the float function because PReLU is positively
homogeneous. Safe GTCRN locations include pointwise-1/PReLU/depthwise and
depthwise/PReLU/pointwise-2 within each temporal block, plus the two simple
stem/decoder pairs. Do not carry such rescaling across attention energy,
sigmoid, LayerNorm or an uncompensated skip branch. Grouped and transpose
weight indexing needs its own float-equivalence test. This is a known
quantization reparameterization principle, not a new architecture technique.
[Original equalization paper](https://arxiv.org/abs/1906.04721).

The complete GPU QAT trainer should then:

1. Load the selected normalized float checkpoint and the chosen audited
   grids. Freeze BatchNorm running statistics and fold them into the trainable
   convolution parameterization. Match the converted kernels and all 188
   boundaries; preserve the fixed integer histories in the forward pass.
2. Implement trainable affine/PReLU operators with power-of-two weight grids,
   INT32 bias rounding and the exact current output rounding. Save activation
   grids, weight-grid policy and all nonlinear/normalization settings in the
   checkpoint; exporter reconstruction must use that saved policy.
3. Replace per-forward CPU parameter snapshots in the current numerical QAT
   prototypes. Prepare or update quantization metadata on-device. Batch GRU
   input projections across time/frequency, retain the necessary recurrent
   loop, and use tensor INT64 only where the proven intermediate bounds
   require it. Keep exact integer forward values with a documented float
   gradient surrogate; freezing a recurrent weight does not remove the need
   for gradients through its operation to earlier convolutions.
4. Start with fixed activation grids. Any later learned-range experiment
   must project scales to the existing power-of-two set and freeze them
   before evaluating a stream. Unconstrained real-valued learned step sizes
   require different deployment scale metadata/arithmetic and cannot simply
   be exported as current powers of two. Learned-step-size research supports
   optimizing quantizers, but its image-classification gains do not predict
   this denoiser's gain. [LSQ paper](https://arxiv.org/abs/1902.08153).
5. Match training mixture, clean exposure and level loss to a float
   continuation control. Compare QAT and float from the same initial
   checkpoint and budget. Select on declared development criteria; require
   full packed C/PCM16 quality, clean preservation and exact integer parity.

An implementation opportunity is explicit integer-code affine work in FP32
on the GPU: both frozen checkpoints' validated maximum absolute affine
accumulator bounds are far below 2^24 (227,010 raw; 270,064 normalized).
Direct INT8-code products and every partial sum within that bound are exactly
representable in FP32. A gather/unfold plus matrix-multiply path can exploit
that fact while retaining an exact integer forward reference. This argument
does not automatically cover Winograd/FFT convolution transforms or relaxed
kernel arithmetic, and QAT can change the bounds. Validate the actual GPU
backend and check bounds after parameter updates; disable approximate modes
for the acceptance oracle. PyTorch documents separate precision controls for
matrix multiplication and convolution. [Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html),
[CUDA precision controls](https://docs.pytorch.org/docs/main/notes/cuda.html).

No GPU QAT throughput, recovery amount, physical MCU latency or new quality
improvement is claimed by this plan or the MSE implementation.
