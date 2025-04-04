#!/bin/bash

if ! [ $# -eq 1 ]; then
    echo "Provide backend as command-line argument"
    exit 1
fi

julia --project=JACC -e 'import Pkg; Pkg.instantiate()'
julia --project=JACC -e 'import JACC; JACC.set_backend(ARGS[1])' $1
