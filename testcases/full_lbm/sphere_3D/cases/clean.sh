#!/bin/bash


for subdir in Re*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done
