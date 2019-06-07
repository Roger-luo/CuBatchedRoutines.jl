module CuBatchedRoutines

using GPUArrays, CuArrays
using Reexport
@reexport using BatchedRoutines
import BatchedRoutines: batched_gemm!, batched_gemm, batched_tr!, batched_tr

include("blas.jl")
include("linalg.jl")

end # module
