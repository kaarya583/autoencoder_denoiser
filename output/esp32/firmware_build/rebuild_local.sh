#!/bin/bash
set -euo pipefail
export IDF_TOOLS_PATH=/private/tmp/esp32-denoiser-idf-tools-v5.4.2
export SSL_CERT_FILE=/etc/ssl/cert.pem
export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:/usr/bin:/bin
source /private/tmp/esp32-denoiser-idf-v5.4.2/export.sh
cd /Users/sidchat/Documents/GitHub/autoencoder_denoiser/firmware/esp32_benchmark
idf.py -B /private/tmp/esp32-denoiser-build-v5.4.2 -D SDKCONFIG=/tmp/esp32-denoiser-perf-sdkconfig build
idf.py -B /private/tmp/esp32-denoiser-build-v5.4.2 size
