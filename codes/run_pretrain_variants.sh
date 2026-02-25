#!/bin/bash
# Training script for SymUNet Pretrain variants
# Usage: bash run_pretrain_variants.sh [dry-run]

MANIFEST="tools/manifest_pretrain_variants.json"

if [ "$1" == "dry-run" ]; then
    echo "=== DRY RUN MODE ==="
    python3 tools/run_batch.py --manifest $MANIFEST --dry-run
else
    echo "=== START TRAINING ==="
    python3 tools/run_batch.py --manifest $MANIFEST
fi
