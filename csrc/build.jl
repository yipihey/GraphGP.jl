# Build the optional custom-CUDA accelerator library (libgraphgpcapi.so) from graphgp_capi.cu.
#
#   julia csrc/build.jl                     # auto-detects nvcc + GPU arch
#   NVCC=/path/to/nvcc GPU_ARCH=sm_86 julia csrc/build.jl
#
# This is OPTIONAL and NOT part of the package build/CI: GraphGP.jl runs fully on the portable
# KernelAbstractions path without it. When the .so is present (and CUDA is loaded), the custom path
# can be selected with `refine_logdet(prob; custom = true)` etc. The resulting library path is
# printed; point GraphGP at it with `ENV["GRAPHGP_CUDA_LIB"]` or rely on the default location below.

const HERE = @__DIR__
const OUT = joinpath(HERE, "libgraphgpcapi.so")

function find_nvcc()
    haskey(ENV, "NVCC") && return ENV["NVCC"]
    cands = [
        # the cu13 toolkit graphgp_cuda itself was built with (pip nvidia-cuda-nvcc)
        joinpath(@__DIR__, "..", ".venv-gpu", "lib", "python3.11", "site-packages",
                 "nvidia", "cu13", "bin", "nvcc"),
        Sys.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ]
    for c in cands
        c !== nothing && isfile(c) && return c
    end
    error("nvcc not found. Set NVCC=/path/to/nvcc.")
end

const NVCC = find_nvcc()
const ARCH = get(ENV, "GPU_ARCH", "sm_86")   # A6000 = Ampere GA102

cmd = `$NVCC -O3 -std=c++17 --shared -Xcompiler -fPIC
       -gencode arch=compute_$(ARCH[4:end]),code=$ARCH
       -o $OUT $(joinpath(HERE, "graphgp_capi.cu"))`
@info "building custom-CUDA accelerator" nvcc=NVCC arch=ARCH out=OUT
run(cmd)
@info "built" lib=OUT size_kb=round(filesize(OUT) / 1024; digits=1)
println(OUT)
