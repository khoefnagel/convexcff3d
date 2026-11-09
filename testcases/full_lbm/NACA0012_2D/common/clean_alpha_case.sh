#!/bin/bash

for subdir in c*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done

