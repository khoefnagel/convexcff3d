#!/bin/bash


for subdir in R*/; do
	cd "$subdir"
	python postprocess.py
	cd ../
done
