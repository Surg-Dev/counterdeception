import signal

from algo import build_stiener_seed, compute_metric, compute_tree
from util import *
from bruteforce import bruteforce

def sprint_benchmark(factory):
    # Set a time parameter t, run on the gamet of possible times, and run algo + rand gen
    # Interrupt each when time runs out and take the last complete best tree.
    G, s, targets, budget = factory()

    rand_best = None
    algo_best = None
    rand_halt = False
    algo_halt = False

    def random_bench_internal(G, s, targets, budget):
        nonlocal rand_best
        # Build n random spanning trees over G, compute metric, take max
        rand_best = float("-inf")
        attempts = []
        i = 0
        while not rand_halt:
            i+=1
            print(
                f"Generating Random Spanning Tree {bcolors.OKGREEN}{i + 1}{bcolors.ENDC}"
            )
            size = float("inf")

            attempt_count = 0
            while size > budget or size == float("inf"):  # TODO: Add failsafe here
                rst, _ = build_stiener_seed(G, s, targets, minimum=None)
                size = rst.size(weight="weight")
                attempt_count += 1
            forced, metric, _ = compute_metric(rst, s, targets)
            if rand_halt:
                break
            res = metric if not forced else 0.0
            if res > rand_best:
                rand_best = res
            print(bcolors.CLEAR_LAST_LINE)
            attempts.append(attempt_count)
    
    def algo_bench_internal(G, s, targets, budget):
        nonlocal algo_best
        algo_best = float("-inf")
        i = 0
        while not algo_halt:
            i += 1
            print(
                f"Computing Algo Tree {bcolors.OKGREEN}{i + 1}{bcolors.ENDC}"
            )
            mst, pred, _ = compute_tree(G, s, targets, budget, minimum=None)
            # We don't cancel the algorithms current run, but if we halted during the run, dont update
            if algo_halt:
                break
            forced, metric, _ = compute_metric(mst, s, targets, pred)
            curr_algo_res = metric if not forced else 0.0
            algo_best = max(algo_best, curr_algo_res)
            print(bcolors.CLEAR_LAST_LINE)
    
    def rand_timeout(signum, frame):
        nonlocal rand_halt
        rand_halt = True
    
    def algo_timeout(signum, frame):
        nonlocal algo_halt
        algo_halt = True

    timestamp = 20

    print(f"Starting Sprint Benchmark, current time limit is {timestamp} seconds.")
    
    signal.signal(signal.SIGALRM, rand_timeout)
    signal.alarm(timestamp)
    random_bench_internal(G, s, targets, budget)
    signal.alarm(0)

    signal.signal(signal.SIGALRM, algo_timeout)
    signal.alarm(timestamp)
    algo_bench_internal(G, s, targets, budget)
    signal.alarm(0)

    return rand_best, algo_best