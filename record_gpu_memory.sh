#!/bin/bash

# Name of the output CSV file
OUTPUT_FILE="gpu_memory_usage.csv"

# Write the CSV header
echo "Timestamp, GPU Memory Usage (MiB)" > $OUTPUT_FILE

# Run the loop for 60 minutes (3600 seconds)
END=$((SECONDS+3600))

while [ $SECONDS -lt $END ]; do
    # Extract GPU memory usage and timestamp, then append to the CSV
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | while IFS= read -r memory; do
        echo "$(date +'%Y-%m-%d %H:%M:%S'), $memory" >> $OUTPUT_FILE
    done
    # Wait for 1 second before the next iteration
    sleep 1
done
