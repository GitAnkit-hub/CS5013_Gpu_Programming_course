#!/bin/bash

# =============================================================================
# BANG Exact Distance - DEEP100M Dataset Run Script
# =============================================================================

# Data path - UPDATE THIS TO YOUR DATA LOCATION
DATA_PATH="/mnt/hdd_volume2/deep100m96"

# Parameters
NUM_QUERIES=10000
K4_THREADS=1
K1_THREADS=256
K2_THREADS=512
K5_THREADS=256
RECALL_AT=10
CPU_THREADS=64
CAPABILITIES=0

echo "=============================================="
echo "BANG Exact Distance - DEEP100M Optimized"
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

# Run the executable
./bang_exact \
    "${DATA_PATH}/deep100m_pq_pivots.bin" \
    "${DATA_PATH}/deep100m_pq_compressed.bin" \
    "${DATA_PATH}/deep100m_graph.bin" \
    "${DATA_PATH}/deep100m_query.bin" \
    "${DATA_PATH}/deep100m_chunk_offsets.bin" \
    "${DATA_PATH}/deep100m_centroid.bin" \
    "${DATA_PATH}/deep100m_gndtruth.bin" \
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
