#!/bin/bash


for subdir in R*/; do
	cd "$subdir"
	python run.py
	python postprocess.py
	cd ../
done

