#!/bin/sh

# Script used to convert jpg files to png files for our dataset.

shopt -s nullglob

files=( *.JPG *.jpg )
if [ ${#files[@]} -eq 0 ]; then
    echo "No JPG files found in the current directory."
    exit 1
fi

readarray -t sorted_files < <(printf '%s\n' "${files[@]}" | sort)

counter=1
for file in "${sorted_files[@]}"; do
    output="${counter}.png"
    echo "Converting $file -> $output"
    ffmpeg -i "$file" -y "$output"
    ((counter++))
done

echo "Done. Converted $((counter-1)) files."
