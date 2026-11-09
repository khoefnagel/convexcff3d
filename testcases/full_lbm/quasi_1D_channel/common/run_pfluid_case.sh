#!/bin/bash

for subdir in N*/; do
	cd "$subdir"
	python run.py
	python postprocess.py
	cd ../
done
