#!/bin/bash

# =============================================================================
# BANG Exact Distance - SIFT100M Dataset Run Script
# =============================================================================

# Data path - UPDATE THIS TO YOUR DATA LOCATION
DATA_PATH="/mnt/karthik_hdd_4tb/sift100m128"

# Parameters
NUM_QUERIES=10000
K4_THREADS=1
K1_THREADS=256
K2_THREADS=512
K5_THREADS=256
RECALL_AT=10
CPU_THREADS=64
CAPABILITIES=1

echo "=============================================="
echo "BANG Exact Distance - SIFT100M Optimized"
echo "=============================================="
echo "Data Path: $DATA_PATH"
echo "Queries: $NUM_QUERIES"
echo "Recall@: $RECALL_AT"
echo "=============================================="
echo ""

# Check if executable exists
if [ ! -f "./bang_exact" ]; then
    echo "Error: bang_exact executable not found!"
    echo "Please run 'make' first."
    exit 1
fi

# Check if data files exist
FILES=(
    "DiskANNsift100m64_pq_pivots.bin"
    "DiskANNsift100m64_pq_compressed.bin"
    "sift100m_disk.index"
    "sift100m_query.bin"
    "DiskANNsift100m64_pq_pivots.bin_chunk_offsets.bin"
    "DiskANNsift100m64_pq_pivots.bin_centroid.bin"
    "sift100m_gndtruth.bin"
)

for f in "${FILES[@]}"; do
    if [ ! -f "${DATA_PATH}/${f}" ]; then
        echo "Warning: ${DATA_PATH}/${f} not found"
    fi
done

# Run the executable
./bang_exact \
    "${DATA_PATH}/DiskANNsift100m64_pq_pivots.bin" \
    "${DATA_PATH}/DiskANNsift100m64_pq_compressed.bin" \
    "${DATA_PATH}/sift100m_disk.index" \
    "${DATA_PATH}/sift100m_query.bin" \
    "${DATA_PATH}/DiskANNsift100m64_pq_pivots.bin_chunk_offsets.bin" \
    "${DATA_PATH}/DiskANNsift100m64_pq_pivots.bin_centroid.bin" \
    "${DATA_PATH}/sift100m_gndtruth.bin" \
    $NUM_QUERIES \
    $K4_THREADS \
    $K1_THREADS \
    $K2_THREADS \
    $K5_THREADS \
    $RECALL_AT \
    $CPU_THREADS \
    $CAPABILITIES

echo ""
echo "=============================================="
echo "Execution completed."
echo "=============================================="
