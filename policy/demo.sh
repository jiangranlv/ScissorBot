#!/bin/bash
while true
do
    python policy/generate_demos_high_level.py setup.cuda=$1 
done