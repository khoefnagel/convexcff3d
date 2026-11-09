#!/bin/bash

for subdir in theta*/; do
	cd "$subdir"
	./postprocess.sh
	cd ../
done
