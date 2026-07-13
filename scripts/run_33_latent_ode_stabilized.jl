# Process coordinator for scripts/33_latent_ode_stabilized.jl.
#
# Production (default): serial prep/deployed, at most three isolated ensemble member
# children, then single-writer finalize. Smoke uses separate checkpoints, two tiny
# members, explicitly exercises one completed-member resume, and writes no production
# aggregate/model/figure.

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
const WORKER_SCRIPT = joinpath(@__DIR__, "33_latent_ode_stabilized.jl")
const SMOKE = get(ENV, "LATENT33_SMOKE", "0") == "1"
const MAX_WORKERS = 3
const CHECKPOINT_ROOT = joinpath(
    PROJECT_DIR, "data",
    SMOKE ? "latent_ode_stabilized_smoke" : "latent_ode_stabilized")
const MANIFEST_PATH = joinpath(CHECKPOINT_ROOT, "manifest.jld2")

import Pkg
Pkg.activate(PROJECT_DIR; io = devnull)
using JLD2

function child_command(phase; member = 0)
    base = `$(Base.julia_cmd()) --project=$(PROJECT_DIR) $(WORKER_SCRIPT)`
    return addenv(base,
                  "LATENT33_PHASE" => phase,
                  "LATENT33_SMOKE" => (SMOKE ? "1" : "0"),
                  "LATENT33_MEMBER" => string(member),
                  "JULIA_NUM_THREADS" => "1",
                  "OPENBLAS_NUM_THREADS" => "3",
                  "OMP_NUM_THREADS" => "1",
                  "MKL_NUM_THREADS" => "1")
end

function run_serial(phase; member = 0)
    println("COORDINATOR_START phase=$phase member=$member smoke=$SMOKE")
    flush(stdout)
    run(child_command(phase; member))
    println("COORDINATOR_DONE phase=$phase member=$member")
    flush(stdout)
end

function monitor_member!(completions, monitors, member, process)
    monitors[member] = @async begin
        failure = try
            wait(process)
            success(process) ? nothing :
            ErrorException("ensemble member $member failed")
        catch error
            error
        end
        status = failure === nothing ? "DONE" : "FAILED"
        println("COORDINATOR_MEMBER_$status member=$member")
        flush(stdout)
        put!(completions, (member, failure))
    end
end

function run_member_pool(nens)
    active = Dict{Int,Base.Process}()
    monitors = Dict{Int,Task}()
    # One completion per registered member makes monitor puts non-blocking even
    # while the coordinator is handling a different member's failure.
    completions = Channel{Tuple{Int,Any}}(nens)
    next_member = 1
    try
        while next_member <= nens || !isempty(active)
            while next_member <= nens && length(active) < MAX_WORKERS
                println("COORDINATOR_MEMBER_START member=$next_member")
                flush(stdout)
                process = run(
                    child_command("member"; member = next_member); wait = false)
                active[next_member] = process
                monitor_member!(completions, monitors, next_member, process)
                next_member += 1
            end
            member, failure = take!(completions)
            delete!(active, member)
            failure === nothing || throw(failure)
        end
    catch
        for process in values(active)
            process_exited(process) || kill(process)
        end
        foreach(wait, values(active))
        # Every monitor prints its terminal status before completing. Waiting here
        # preserves DONE lines for siblings that finished in the same failure batch.
        foreach(wait, values(monitors))
        rethrow()
    end
end

function main()
    run_serial("prep")
    isfile(MANIFEST_PATH) || error("prep did not write manifest: $MANIFEST_PATH")
    manifest = load(MANIFEST_PATH)
    nens = manifest["nens"]
    expected = SMOKE ? 2 : (nens in (5, 20) ? nens :
                            error("production NENS must be 5 or 20, got $nens"))
    nens == expected || error("unexpected smoke NENS=$nens")
    println("COORDINATOR_POOL nens=$nens max_workers=$MAX_WORKERS")
    flush(stdout)
    run_member_pool(nens)
    if SMOKE
        # The second invocation must load and accept the completed matching checkpoint.
        run_serial("member"; member = 1)
    end
    run_serial("finalize")
    println(SMOKE ? "COORDINATOR_SMOKE_PASS" : "COORDINATOR_PRODUCTION_DONE")
    flush(stdout)
end

main()
