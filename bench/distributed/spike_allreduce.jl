# Phase 0 plumbing spike: MPI rank/GPU binding + Allreduce (host and, if available, device).
#
#   mpiexec -n <R> julia --project=julia/GraphGP/bench/distributed spike_allreduce.jl
#
# Verifies: MPI.Init, one-rank-per-GPU binding via the node-local communicator, a host
# Allreduce, and a CUDA-aware-or-host-staged device Allreduce. This is the gate before any
# GraphGP-specific distributed code: if this runs across nodes under srun, the rest follows.
using MPI
using CUDA

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Node-local rank → bind one GPU per rank (before any CUDA allocation).
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    local_rank = MPI.Comm_rank(local_comm)

    gpu_msg = "no CUDA"
    have_cuda_dev = false
    if CUDA.functional()
        ndev = length(CUDA.devices())
        CUDA.device!(local_rank % ndev)
        have_cuda_dev = true
        gpu_msg = "bound GPU $(local_rank % ndev) of $ndev (dev=$(CUDA.device()))"
    end

    # Host Allreduce.
    host_sum = MPI.Allreduce(Float64(rank + 1), MPI.SUM, comm)
    host_expected = nranks * (nranks + 1) / 2

    # Device Allreduce (CUDA-aware if MPI.has_cuda(), else host-staged).
    dev_msg = "skipped (no GPU)"
    if have_cuda_dev
        v = CUDA.fill(Float64(rank + 1), 4)
        if MPI.has_cuda()
            MPI.Allreduce!(v, MPI.SUM, comm)                 # GPUDirect
            dev_msg = "CUDA-aware Allreduce → $(Array(v)[1]) (has_cuda=true)"
        else
            h = Array(v)                                     # host-stage fallback (tiny payload)
            MPI.Allreduce!(h, MPI.SUM, comm)
            copyto!(v, h)
            dev_msg = "host-staged Allreduce → $(Array(v)[1]) (has_cuda=false)"
        end
    end

    for r in 0:(nranks - 1)
        if r == rank
            ok = isapprox(host_sum, host_expected)
            println("rank $rank/$nranks  local_rank=$local_rank  $gpu_msg")
            println("    host Allreduce = $host_sum (expected $host_expected) $(ok ? "OK" : "FAIL")")
            println("    device: $dev_msg")
            flush(stdout)
        end
        MPI.Barrier(comm)
    end
    rank == 0 && println("\nspike OK: $nranks ranks, MPI.has_cuda()=$(MPI.has_cuda())")
    MPI.Finalize()
end

main()
