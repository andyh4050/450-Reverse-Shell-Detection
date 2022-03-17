#!/usr/bin/bash

for i in {1..10}
do
	echo "running experiment no.$i"
	python test.py &>> log.txt
done
