#!/bin/bash
python3 rotate_detect.py --method MSP --num_eval 5
python3 rotate_detect.py --method MaxMax --num_eval 5
python3 rotate_detect.py --method MeanMax --num_eval 5
python3 rotate_detect.py --method MeanPosMax --num_eval 5
python3 rotate_detect.py --method JSD --num_eval 5
