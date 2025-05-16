#!/bin/bash

SRC_DIR="$1"
TRGT_DIR="$2"

echo "Seperating images by file name into their own directories. This can take a while..."

# Loop through all files in the source directory
i=0
for file in "$SRC_DIR"/*; do
    # Extract the class number (everything before the first '-')
    class_number=$(basename "$file" | grep -oE '^[0-9]+')

    # Skip files that don't match the pattern
    if [[ -z "$class_number" ]]; then
        continue
    fi

    # Create the target directory named after the class number
    target_dir="$TRGT_DIR/$class_number"
    mkdir -p "$target_dir"

    # Copy the file into the target directory
    cp "$file" "$target_dir"
    rm "$file"

    i=$(($i+1))
done

echo "$i files have been organized into class directories and removed from the old directory."
