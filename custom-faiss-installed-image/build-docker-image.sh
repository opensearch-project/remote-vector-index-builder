#!/bin/bash

# Exit on any error
set -xe

# Function to check if command was successful
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Function to check if patch file exists
check_patch_file() {
    if [ ! -f "$1" ]; then
        echo "Error: Patch file '$1' not found!"
        exit 1
    fi
}

apply_patch() {
    echo "Updating FAISS submodule..."
    git submodule update --init -- faiss
    check_status "Failed to update submodule"

    # Check if patch file is provided as argument
    if [ $# -ne 1 ]; then
        echo "Usage: $0 "
        exit 1
    fi

    PATCH_FILE="$1"
    check_patch_file "$PATCH_FILE"

    # Navigate into the faiss directory
    echo "Entering FAISS directory..."
    cd faiss || exit 1

    # Apply the patch
    echo "Applying patch..."
    git apply ../"$PATCH_FILE"
    check_status "Failed to apply patch"

    chmod -R 777 .

    echo "Successfully applied patch to FAISS!"
    cd ..
}

apply_patch "0001-added-commit-to-enable-the-add_with_ids-function-on-.patch"

docker build -t custom-faiss:latest .
