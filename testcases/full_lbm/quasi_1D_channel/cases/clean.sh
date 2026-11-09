#!/bin/bash

for subdir in pfluid*-*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done

rm -f Figures/*
rm -f orderOfConvergence.txt
