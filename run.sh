#!/bin/bash
echo "Welcome to my submission!"
echo "Please enter path of survive.db to begin"
read db_path

echo "printing results..."
python3 src/main.py "$db_path"
echo "results.txt saved in same directory"
