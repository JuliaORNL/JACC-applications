#!/bin/bash
salloc -A TRN047 -J jacc_test_$USER -t 30:00 -p batch -N 1
