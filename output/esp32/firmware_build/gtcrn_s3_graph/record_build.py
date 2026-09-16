"""Record a completed, source-stable ESP32-S3 build without claiming a board run."""
from pathlib import Path
import hashlib,json,shutil,subprocess,datetime
from elftools.elf.elffile import ELFFile
root=Path.cwd();out=root/'output/esp32/firmware_build/gtcrn_s3_graph';build=Path('/private/tmp/gtcrn-denoiser-build-v5.4.2')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((out/'source_hashes_before.json').read_text());after={name:sha(root/name) for name in before}
assert before==after,'Source changed during the final build'
(out/'source_hashes_after.json').write_text(json.dumps(after,indent=2)+'\n')
association=json.loads((out/'model_association.json').read_text())
for name,digest in [('binary','binary_sha256'),('source_checkpoint','source_checkpoint_sha256'),('calibration_sidecar','calibration_sidecar_file_sha256')]:
 assert sha(root/association[name])==association[digest]
artifacts={}
for name in ['gtcrn_integer_benchmark.bin','gtcrn_integer_benchmark.elf','gtcrn_integer_benchmark.map','compile_commands.json','project_description.json','flasher_args.json','config/sdkconfig.json','partition_table/partition-table.bin','bootloader/bootloader.bin']:
 destination=out/'build'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(build/name,destination)
 artifacts[name]={'bytes':destination.stat().st_size,'sha256':sha(destination)}
shutil.copy2('/private/tmp/gtcrn-denoiser-sdkconfig',out/'build/sdkconfig')
shutil.copy2(root/association['binary'],out/'build/model_int8.bin')
for base in ['experimental_gtcrn','main']:
 for source in (build/'esp-idf'/base).rglob('*.su'):
  destination=out/'stack_usage'/source.name;destination.parent.mkdir(exist_ok=True);shutil.copy2(source,destination)
with (build/'gtcrn_integer_benchmark.elf').open('rb') as f:
 elf=ELFFile(f);symbols={s.name:s for s in elf.get_section_by_name('.symtab').iter_symbols()}
 first=symbols['gtcrn_model_start'];start=first['st_value'];end=symbols['gtcrn_model_end']['st_value'];section=elf.get_section(first['st_shndx'])
 embedded=section.data()[start-section['sh_addr']:end-section['sh_addr']]
 assert start%16==0 and embedded==(root/association['binary']).read_bytes()
 undefined=[name for name,s in symbols.items() if s['st_shndx']=='SHN_UNDEF' and name]
 assert not [name for name in undefined if name.startswith(('edng_','ednx_','dsps_'))]
 sections={section.name:section['sh_size'] for section in elf.iter_sections()}
 fft_symbols={name:hex(symbols[name]['st_value']) for name in ['dsps_fft2r_fc32_aes3_','dsps_bit_rev_fc32_ansi','dsps_fft2r_w_table_fc32_1024']}
 memory={'model_handle_bytes':5784,'neural_state_bytes':18048,'neural_workspace_bytes':17536,'audio_state_bytes':12528,'explicit_runtime_bytes':53896,'runtime_arena_alignment_padding':8,'allocated_internal_arena_bytes':53904,'pcm_buffers_bytes':1024,'fft_bit_reversal_table_heap_request_bytes':960,'main_task_reserved_stack_bytes':8192,'benchmark_timing_samples_bytes':4096}
 memory['explicit_audio_runtime_stack_fft_pcm_bytes']=53904+1024+960+8192
 memory['benchmark_subtotal_model_in_flash_bytes']=memory['explicit_audio_runtime_stack_fft_pcm_bytes']+4096
 memory['benchmark_subtotal_model_copied_to_sram_bytes']=memory['benchmark_subtotal_model_in_flash_bytes']+len(embedded)
 memory['whole_application_static_dram_writable_bytes']=sections['.dram0.data']+sections['.dram0.bss']
 memory['whole_application_static_rtc_bytes']=sections['.rtc.force_fast']+sections['.rtc_reserved']
 memory['accounting_note']='Explicit requests/reservations, not observed heap/stack peaks. Application static DRAM includes PCM and timing arrays; do not add those twice. Float PCM scratch is inside the reserved task stack. ROM-provided S3 FFT twiddle table has no application heap request. Allocator headers, other RTOS tasks, interrupt stacks, ROM/system reservations, I2S/DMA, radio and integrated application features are outside the ML subtotal.'
config=json.loads((build/'config/sdkconfig.json').read_text())
assert config['IDF_TARGET']=='esp32s3' and config['ESP_DEFAULT_CPU_FREQ_MHZ']==240 and not config['SPIRAM']
commands=json.loads((build/'compile_commands.json').read_text())
selected=[entry for entry in commands if '/experimental_gtcrn/' in entry['file'] or '/experimental_int8/' in entry['file'] or '/experimental_dsp/' in entry['file'] or '/gtcrn_benchmark/main/' in entry['file']]
(out/'selected_compile_commands.json').write_text(json.dumps(selected,indent=2)+'\n')
compiler=selected[0]['command'].split()[0]
report={'status':'ESP32-S3 compile/link and host reference checks passed; hardware execution pending','recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':association,'toolchain':{'sdk':'ESP-IDF5.4.2','dsp':'ESP-DSP1.8.2','compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],'compiler_flags':['-O2','-std=gnu17','-mlongcalls','-fstack-usage'],'target':'esp32s3','cpu_mhz':240,'psram_enabled':False,'flash_mode':config['ESPTOOLPY_FLASHMODE'],'flash_mhz':80,'flash_size':config['ESPTOOLPY_FLASHSIZE']},'app_bin_bytes':artifacts['gtcrn_integer_benchmark.bin']['bytes'],'factory_partition_bytes':1048576,'factory_partition_free_bytes':1048576-artifacts['gtcrn_integer_benchmark.bin']['bytes'],'embedded_model':{'bytes':len(embedded),'sha256':hashlib.sha256(embedded).hexdigest(),'address':hex(start),'alignment_bytes':16,'placement':'flash_mapped'},'memory':memory,'linked_fft_symbols':fft_symbols,'validation':{'source_before_after_hashes_equal':True,'source_files':len(before),'packed_calibration_and_checkpoint_association_verified':True,'embedded_blob_byte_exact':True,'elf_no_unresolved_gtcrn_or_dsp_symbols':True,'sdk_undefined_symbol_entries':undefined,'host_numpy_c_known_answer_frames':12,'known_answer_checks_entire_mask_and_state_on_host':True,'onboard_known_answer_execution':False,'board_flashed':False,'latency_measured':False,'peak_heap_or_stack_measured':False},'artifacts':artifacts,'limitations':['The trained candidate is a development probe, not final selected model.','Packed model bytes exclude executable firmware, vendor FFT data and RAM.','Portable scalar integer kernels are linked; only the FFT uses ESP-DSP S3 optimization.','No real-time, end-to-end I2S or on-device quality claim follows from this build.','The optional internal-SRAM copy is an accounting projection; only default flash placement was built.']}
(out/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'app_bin_bytes':report['app_bin_bytes'],'packed_bytes':len(embedded),'memory':memory,'source_files':len(before)},indent=2))
