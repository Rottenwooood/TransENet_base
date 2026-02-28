#!/bin/bash

prefix="$1"

if [ -z "$prefix" ]; then
    echo "用法: $0 <前缀>"
    exit 1
fi

for file in model/${prefix}*.py; do
    [ -e "$file" ] || continue
    new_name=$(echo "$file" | tr '[:upper:]' '[:lower:]')
    if [ "$file" != "$new_name" ]; then
        echo "重命名: $file -> $new_name"
        mv "$file" "$new_name"
    fi
done