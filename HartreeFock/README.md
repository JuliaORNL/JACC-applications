# To run the benchmark example
```shell
Threads:
export JULIA_EXCLUSIVE=1
export JULIA_NUM_THREADS=#
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=.
using JACC
JACC.set_backend("threads")
exit()

julia --project=.
using JACC
@show JACC.backend # should show "threads"
exit()

julia --project=. benchmark.jl

GPUs:
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=.
using JACC
JACC.set_backend("cuda")
exit()

julia --project=.
using JACC
@show JACC.backend # should show "cuda"
exit()

julia --project=. benchmark.jl

```
