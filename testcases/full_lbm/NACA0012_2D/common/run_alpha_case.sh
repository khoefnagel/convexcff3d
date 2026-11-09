#!/bin/bash

for subdir in c*/; do
	cd "$subdir"
	python run.py
	cd ../
done
