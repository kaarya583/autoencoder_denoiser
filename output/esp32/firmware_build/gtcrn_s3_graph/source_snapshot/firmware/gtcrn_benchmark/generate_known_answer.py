"""Generate binary-bound startup hashes after exact NumPy/host-C agreement."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from esp32_denoiser.gtcrn_integer_export import load_gtcrn_integer
from esp32_denoiser.gtcrn_native import CIntegerGTCRN, STATE_LAYOUT


def fnv1a(data):
    result = 2166136261
    for value in data:
        result = ((result ^ value) * 16777619) & 0xFFFFFFFF
    return result


def generate(source: Path, destination: Path):
    blob = source.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    reference, native = load_gtcrn_integer(blob), CIntegerGTCRN(blob)
    random, native_state, reference_state, hashes = 2026, None, None, []
    for frame in range(12):
        features = np.zeros(387, np.int8)
        if frame:
            for index in range(387):
                random ^= (random << 13) & 0xFFFFFFFF
                random ^= random >> 17
                random ^= (random << 5) & 0xFFFFFFFF
                features[index] = (random & 255) - 128
        mask, native_state = native.neural_step(features, native_state)
        real = features.astype(np.float32).reshape(1, 3, 1, 129) * reference.backend.grid("erb.output").scale
        expected, reference_state = reference.graph.frame(real, reference_state)
        expected_state = np.concatenate([reference_state[name].reshape(-1) for name, _, _, _ in STATE_LAYOUT])
        np.testing.assert_array_equal(mask, expected.data.reshape(2, 129))
        np.testing.assert_array_equal(native_state, expected_state)
        hashes.append((fnv1a(mask.tobytes()), fnv1a(native_state.tobytes())))
    text = "/* Generated after exact NumPy/host-C agreement; startup hashes are a smoke check. */\n"
    text += f'#define GTCRN_KNOWN_ANSWER_MODEL_SHA256 "{digest}"\n'
    text += '#define GTCRN_KNOWN_ANSWER_FRAMES 12\n'
    text += 'static const uint32_t gtcrn_known_answer[12][2] = {\n'
    text += ''.join(f'    {{UINT32_C({mask}), UINT32_C({state})}},\n' for mask, state in hashes)
    text += '};\n'
    if source.read_bytes() != blob:
        raise ValueError("Model changed while generating its known answers")
    destination.write_text(text)
    print(f"Wrote {destination}: 12 exact NumPy/C frame and state matches; model SHA256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "main/model.bin")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "main/known_answer.h")
    args = parser.parse_args()
    generate(args.model, args.output)
