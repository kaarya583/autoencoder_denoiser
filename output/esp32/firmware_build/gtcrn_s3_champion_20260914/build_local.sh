#!/bin/bash
set -euo pipefail
export IDF_TOOLS_PATH=/private/tmp/esp32-denoiser-idf-tools-v5.4.2
export SSL_CERT_FILE=/etc/ssl/cert.pem
export LC_ALL=en_US.UTF-8
export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:/usr/bin:/bin
source /private/tmp/esp32-denoiser-idf-v5.4.2/export.sh
cd "/Users/sidchat/Documents/GitHub/autoencoder_denoiser/output/esp32/firmware_build/gtcrn_s3_champion_20260914/source_snapshot/firmware/gtcrn_benchmark"
idf.py -B /private/tmp/gtcrn-champion-build-20260914 -D SDKCONFIG=/private/tmp/gtcrn-champion-sdkconfig-20260914 -D IDF_TARGET=esp32s3 build
idf.py -B /private/tmp/gtcrn-champion-build-20260914 size
