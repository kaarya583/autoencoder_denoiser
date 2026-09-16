"""Record this isolated champion build using actual target-compiled sizeof data."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import struct
import subprocess
from elftools.elf.elffile import ELFFile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
STAGE = OUT / "source_snapshot"
BUILD = Path("/private/tmp/gtcrn-champion-build-20260914")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
save = lambda p, value: Path(p).write_text(json.dumps(value, indent=2) + "\n")
before = json.loads((OUT / "source_hashes_before.json").read_text())
after = {name: sha(ROOT / name) for name in before}
assert before == after, "Original scientific source or existing benchmark changed"
save(OUT / "source_hashes_after.json", after)
association = json.loads((OUT / "model_association.json").read_text())
assert all(sha(item["path"]) == item["sha256"] for item in association.values())
commands = json.loads((BUILD / "compile_commands.json").read_text())
selected = [entry for entry in commands if str(STAGE / "firmware") in entry["file"] and "/managed_components/" not in entry["file"]]
save(OUT / "selected_compile_commands.json", selected)
probes = {
    "graph": ("graph.c", ["sizeof(edng_model)", "sizeof(edng_operator)", "sizeof(edng_workspace)", "EDNG_NEURAL_STATE_BYTES"]),
    "affine": ("operators.c", ["sizeof(ednx_affine)"]),
    "primitives": ("primitives.c", ["sizeof(ednx_gru)", "sizeof(ednx_layer_norm)"]),
    "audio": ("audio.c", ["sizeof(edng_audio)", "__alignof__(edng_audio)"]),
    "erb": ("gtcrn_erb.c", ["sizeof(ednx_erb)"]),
}
sizes = {}
for name, (filename, expressions) in probes.items():
    entry = next(entry for entry in selected if Path(entry["file"]).name == filename)
    source, obj = OUT / (name + "_sizeof.c"), OUT / (name + "_sizeof.o")
    source.write_text('#include "' + entry["file"] + '"\nconst uint32_t size_probe[] __attribute__((used,section(".size_probe")))={' + ",".join(expressions) + '};\n')
    args = shlex.split(entry["command"])
    args[args.index("-o") + 1] = str(obj)
    args[args.index("-c") + 1] = str(source)
    subprocess.run(args, cwd=entry["directory"], check=True, capture_output=True)
    with obj.open("rb") as handle:
        raw = ELFFile(handle).get_section_by_name(".size_probe").data()
    sizes[name] = dict(zip(expressions, struct.unpack("<" + "I" * len(expressions), raw)))
save(OUT / "target_sizeof.json", sizes)
align8 = lambda n: (n + 7) & ~7
align16 = lambda n: (n + 15) & ~15
# Fixed-topology counts and native handle allocation reproduce packed.c.
model_bytes = (align8(sizes["graph"]["sizeof(edng_model)"]) + 32 * align8(sizes["affine"]["sizeof(ednx_affine)"])
               + 18 * align8(sizes["primitives"]["sizeof(ednx_gru)"]) + 4 * align8(sizes["primitives"]["sizeof(ednx_layer_norm)"]))
audio_bytes = align16(align8(sizes["audio"]["sizeof(edng_audio)"]) + sizes["erb"]["sizeof(ednx_erb)"])
neural_bytes = sizes["graph"]["EDNG_NEURAL_STATE_BYTES"]
workspace_bytes = sizes["graph"]["sizeof(edng_workspace)"]
arena = sum(align16(value) for value in (model_bytes, audio_bytes, neural_bytes, workspace_bytes))
artifacts = {}
for name in ["gtcrn_integer_benchmark.bin", "gtcrn_integer_benchmark.elf", "gtcrn_integer_benchmark.map", "compile_commands.json",
             "project_description.json", "flasher_args.json", "config/sdkconfig.json", "partition_table/partition-table.bin", "bootloader/bootloader.bin"]:
    destination = OUT / "build" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD / name, destination)
    artifacts[name] = dict(bytes=destination.stat().st_size, sha256=sha(destination))
shutil.copy2("/private/tmp/gtcrn-champion-sdkconfig-20260914", OUT / "build/sdkconfig")
shutil.copy2(association["model.bin"]["path"], OUT / "build/model.bin")
for base in ["experimental_gtcrn", "main"]:
    for path in (BUILD / "esp-idf" / base).rglob("*.su"):
        destination = OUT / "stack_usage" / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
with (BUILD / "gtcrn_integer_benchmark.elf").open("rb") as handle:
    elf = ELFFile(handle)
    symbols = {symbol.name: symbol for symbol in elf.get_section_by_name(".symtab").iter_symbols()}
    start, end = symbols["gtcrn_model_start"], symbols["gtcrn_model_end"]
    address, length = start["st_value"], end["st_value"] - start["st_value"]
    section = elf.get_section(start["st_shndx"])
    embedded = section.data()[address - section["sh_addr"]:address - section["sh_addr"] + length]
    assert address % 16 == 0 and embedded == Path(association["model.bin"]["path"]).read_bytes()
    undefined = [name for name, symbol in symbols.items() if name and symbol["st_shndx"] == "SHN_UNDEF"]
    assert not [name for name in undefined if name.startswith(("edng_", "ednx_", "dsps_"))]
    sections = {section.name: section["sh_size"] for section in elf.iter_sections()}
    fft = {name: hex(symbols[name]["st_value"]) for name in ("dsps_fft2r_fc32_aes3_", "dsps_bit_rev_fc32_ansi")}
config = json.loads((BUILD / "config/sdkconfig.json").read_text())
assert config["IDF_TARGET"] == "esp32s3" and config["ESP_DEFAULT_CPU_FREQ_MHZ"] == 240 and not config["SPIRAM"]
stack = config["ESP_MAIN_TASK_STACK_SIZE"]
memory = dict(model_handle_bytes=model_bytes, neural_state_bytes=neural_bytes, neural_workspace_bytes=workspace_bytes,
              audio_state_bytes=audio_bytes, allocated_internal_arena_bytes=arena,
              pcm_buffers_bytes=1024, configured_main_task_stack_bytes=stack, fft_bit_reversal_heap_request_bytes=960,
              benchmark_timing_array_bytes=4096, explicit_audio_reservations_bytes=arena + 1024 + stack + 960,
              explicit_benchmark_subtotal_bytes=arena + 1024 + stack + 960 + 4096,
              whole_application_static_dram_writable_bytes=sections[".dram0.data"] + sections[".dram0.bss"],
              whole_application_static_rtc_bytes=sections[".rtc.force_fast"] + sections[".rtc_reserved"])
report = dict(status="ESP32-S3 champion cross-compile/link and binary-bound host known answers passed",
    recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), model=association,
    toolchain=dict(sdk="ESP-IDF5.4.2", dsp="ESP-DSP1.8.2", target="esp32s3", cpu_mhz=240, psram=False,
                   compiler=subprocess.check_output([shlex.split(selected[0]["command"])[0], "--version"], text=True).splitlines()[0]),
    packed_model=dict(bytes=length, sha256=hashlib.sha256(embedded).hexdigest(), flash_address=hex(address), alignment_bytes=16),
    application_bin_bytes=artifacts["gtcrn_integer_benchmark.bin"]["bytes"],
    factory_partition_bytes=1048576, factory_partition_free_bytes=1048576-artifacts["gtcrn_integer_benchmark.bin"]["bytes"],
    memory=memory, target_sizeof=sizes, linked_fft_symbols=fft, artifacts=artifacts,
    validation=dict(original_source_files_unchanged=len(before), embedded_blob_byte_exact=True,
                    host_numpy_c_known_answer_frames=12, all_masks_and_states_exact=True,
                    board_flashed=False, on_device_known_answers_executed=False, latency_measured=False),
    scope="Current 9.2022137622 dB external-development champion; official test remains sealed. Scalar neural C, ESP-DSP FFT. Compile/link only; no MCU throughput claim.",
    memory_caveats="Target sizeof plus explicit reservations; not observed heap/stack peaks. Whole-app static DRAM overlaps PCM/timing arrays. Excludes allocator headers, other RTOS/interrupt stacks, I2S/DMA/radio and system reservations. PCM float scratch is inside task stack. FFT960B request derives from existing ESP-DSP1.8.2; S3 twiddles are ROM-provided.")
save(OUT / "build_report.json", report)
save(OUT / "artifact_hashes.json", {str(path.relative_to(OUT)): sha(path) for path in sorted(OUT.rglob("*"))
                                  if path.is_file() and "source_snapshot" not in path.parts and path.name != "artifact_hashes.json"})
print(json.dumps(dict(application_bin_bytes=report["application_bin_bytes"], packed_model=report["packed_model"], memory=memory), indent=2))
