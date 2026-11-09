#!/bin/bash


for subdir in Re*/; do
	cd "$subdir"
	./run.sh
	cd ../
done

