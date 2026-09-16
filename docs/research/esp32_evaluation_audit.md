# Full-C evaluation audit

The audited deployment-quality path is `esp32_denoiser.embedded`, which dispatches EDNSI8-v2 binaries to `EmbeddedWaveformEnhancer` and EDNFQ8 binaries to `FrequencyEmbeddedWaveformEnhancer`. There is no local `c_runtime.py` in this repository. `esp32_denoiser.evaluate --integer-model ...` instead uses C neural inference with PyTorch DSP; its result alone is not a measurement of the complete C PCM16 frontend.

## Findings and corrections

The C frontend's alignment is consistent with the trained DSP: 512-sample analysis windows, 256-sample hops, zero left overlap, persistent state across frames, a final zero flush hop, and compensation of the initial output hop. Both neural and DSP histories reset independently for each utterance. Partial final hops are retained. C complex reconstruction handles DC/Nyquist and Hermitian symmetry consistently with a real inverse FFT. The existing neural, host-FFT and ESP-DSP API-stub parity tests pass with nonzero masks and nondefault exported DSP constants. These checks did not identify a shift, future leakage, state leakage, or SI-SDR arithmetic error.

Four reporting/integrity gaps were corrected:

1. **Incomplete audio could be labeled a full utterance.** The dataset reads the declared sample count, so a stale, shorter manifest length previously scored a prefix silently. Evaluation now checks the actual whole-file frame count, sample rate and channel count before inference. The training dataset is unchanged.
2. **Scored bytes were insufficiently bound to the report.** Evaluation now verifies declared `clean_sha256`/`noisy_sha256` fields when present, records both actual audio hashes even for older manifests without these fields, checks file identity/size/timestamps across loading, and rejects a changed manifest. The manifest SHA describes the bytes read at the start of the evaluation. Checkpoint provenance similarly hashes the checkpoint bytes actually deserialized. Full-C comparison passes use one immutable binary byte snapshot and require identical manifest/audio hashes and utterance IDs/lengths. The standalone paired comparator also rejects differing audio hashes when both reports supply them; historical reports remain comparable, with exact matched/unverified audio-utterance counts and an explicit verification scope. Malformed present hashes are rejected instead of being treated as historical omissions.
3. **Level changes were buried in per-utterance records.** The report now also aggregates signed projection gain, RMS ratio, normalized/raw waveform L1, maximum peak, and PCM16 range/rail counts. A regression test shows that multiplying an estimate by −0.25 preserves SI-SDR while the gain, polarity and waveform diagnostics expose the change.
4. **Reusing a frontend could mix PCM counters across suites.** Evaluation reports the increment in its PCM counters for that evaluation only. Full-C CLI metadata uses these scoped counters. A regression test preloads the instance with clipped audio, then verifies that two successive evaluations have identical independent counts.

The metric formulas, datasets, mixing/preparation code, export formats, neural kernels and C DSP were not changed by these corrections. Previously reported numbers are not recalculated or declared invalid by this audit; their reproducibility still depends on retaining the original artifacts and data.

## Interpretation and trust limits

- SI-SDR is centered, scale invariant, polarity invariant, and capped at ±80 dB. It is not a loudness-preservation metric. A good SI-SDR score cannot substitute for the preservation fields or listening. For a clean-input preservation suite, the perfect noisy baseline is capped at 80 dB, so absolute enhanced SI-SDR and the level-sensitive diagnostics are more useful than clean-suite SI-SDRi.
- The mean gives each valid utterance one vote. Silent/DC-only references have undefined centered SI-SDR and are explicitly counted as invalid. Zero output against valid speech is penalized rather than silently omitted. Perceptual metrics have their own reported paired-valid denominators. Comparisons across externally produced reports still need matched valid utterance IDs.
- PCM16 input is rounded to nearest even, clipped to signed 16-bit range, and decoded at 1/32768. The actual C PCM16 output conversion is used. The noisy SI-SDR baseline remains the original manifest waveform, so the final pipeline's score includes its input quantization/clipping cost. No per-model waveform normalization or alignment search is performed.
- `output_at_rail_samples` is a rail-contact count, not proof that all these samples were clipped. The unclipped full-C reference exposes output peaks and samples beyond the PCM16 range. `--compare-reference` compares original-float-input processing with the complete PCM16 path: its difference includes **both input and output** conversion, not output clipping alone. For an isolated output-clipping experiment, feed the same already PCM16-quantized input to both C APIs, as the focused PCM parity test does.
- PESQ/STOI do not establish level preservation. Perceptual preprocessing applies a common attenuation to the clean/noisy/enhanced triplet if required, rather than independently normalizing them; the separate preservation diagnostics retain the original levels.
- The host full-C path uses a portable float32 FFT and scalar host integer kernels. The ESP-DSP stub verifies call ordering and mathematical conventions, not the vendor implementation's numerical behavior on the board. These tests do not execute ESP32-S3 SIMD assembly, measure on-board cycles, prove p99 latency, measure I2S/DMA or radio interaction, or measure heap/stack peaks. RTF is explicitly labeled host offline throughput, excluding compilation, loading, disk I/O and metrics.
- The exported blob size includes its learned integer payload and serialized DSP constants; it excludes executable code, vendor FFT tables, and the rest of firmware. C initialization validates the deployed topology, grids and accumulator bounds. The older global NumPy oracle performs fewer malformed-blob checks than the C loader; it is a numerical reference for trusted exports, not an alternative deployment acceptance check.
- Audio file hashes detect byte changes, not cross-codec copies, historical training leakage, or missing sources in an operator's declared provenance. Manifest/freeze lineage is a separate protocol. No final-test audio was prepared or evaluated during this audit.

## Verification

63 focused tests passed:

```sh
.venv-esp32/bin/python -m pytest \
  tests/test_esp32_evaluate.py tests/test_esp32_embedded.py \
  tests/test_esp32_metric_threads.py tests/test_esp32_frontend.py \
  tests/test_esp32_frequency_frontend.py tests/test_frequency_frontend_contract.py \
  tests/test_integer_runtime.py tests/test_esp32_frequency_runtime.py \
  tests/test_frequency_export.py tests/test_esp32_comparison.py -q
```

These include stale-short/long manifest rejection, declared audio-hash mismatch, manifest mutation, checkpoint replacement during deserialization, binary replacement between reference passes, audio replacement between reference passes, gain/polarity visibility, scoped PCM counters, nontrivial QAT/integer/C waveform parity, final-hop flushing and FFT endpoint behavior.
