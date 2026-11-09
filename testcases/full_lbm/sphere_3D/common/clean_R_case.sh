#!/bin/bash


for subdir in R*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done
