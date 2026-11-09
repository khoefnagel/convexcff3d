#!/bin/bash

for subdir in pfluid*-*/; do
	cd "$subdir"
	./postprocess.sh
	cd ../
done
