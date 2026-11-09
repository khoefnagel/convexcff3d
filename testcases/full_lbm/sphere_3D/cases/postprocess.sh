#!/bin/bash


for subdir in Re*/; do
	cd "$subdir"
	./postprocess.sh
	cd ../
done
