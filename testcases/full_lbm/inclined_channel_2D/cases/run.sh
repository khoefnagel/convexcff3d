#!/bin/bash

for subdir in theta*/; do
	cd "$subdir"
	./run.sh
	./postprocess.sh
	cd ../
done
