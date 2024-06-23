#!/bin/bash

RUN_NAME=living_thing.n.01-proto_box
EXPERIMENT=wsd  # wsd, nsc, hi

python -B -m code.test_$EXPERIMENT \
    --experiment $EXPERIMENT \
    --run_name $RUN_NAME
