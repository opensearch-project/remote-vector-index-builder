#!/bin/bash

# Set Exit on any error
set -xe

update_faiss_submodule() {
    echo "Updating FAISS submodule..."
    git submodule update --init -- faiss

    # Check if command was successful
     if [ $? -ne 0 ]; then
        echo "Error: Failed to update faiss submodule"
        exit 1
    fi

    chmod -R 777 .

    echo "Successfully updated FAISS submodule!"
}

update_faiss_submodule

docker build -t custom-faiss:latest .
