#!/bin/bash

for subdir in N*/; do
	cd "$subdir"
	python postprocess.py
	cd ../
done
