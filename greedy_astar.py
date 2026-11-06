"""
Find the shortest route to buy all ingredients, AND RETURN HOME AFTER SHOPPING.

  - Greedy: always go to the nearest place that still sells at least one missing ingredient.
  - A*    : state = (position, set of missing ingredients), admissible heuristic
            h(p, S) = d(p, Home)                                         if S is empty
            h(p, S) = max_{i in S} min_{m sells i} [d(p,m) + d(m,Home)]  if S is not empty

Usage:
    python greedy_astar.py                              # all inputs in markets_input.json
    python greedy_astar.py markets_input.json input2    # a single input
    python greedy_astar.py markets_input.json --repeat 50   # change the number of repeats used for timing

For each input, run() prints 3 groups of criteria for Greedy and A*:
    - Average execution time (seconds)
    - Peak memory usage (bytes)
    - Route optimality (steps, distance, compared with the shortest route)
"""
import argparse
import heapq
import itertools
import json
import math
import time
import tracemalloc

HOME = -1

def load_input(path, key):
    """Read one input from the JSON file -> (home coordinates, list of places, ingredients to buy)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)[key]
    home = tuple(data["start"]["pos"])
    markets = [
        {"name": m["name"], "pos": tuple(m["pos"]), "items": frozenset(m["items"])}
        for m in data["markets"]
    ]
    need = frozenset(data["shopping_list"])
    return home, markets, need


def dist(a, b):
    return math.dist(a, b)


def check_feasible(markets, need):
    available = set().union(*(m["items"] for m in markets))
    missing = need - available
    if missing:
        raise ValueError(f"No place sells: {sorted(missing)}")


def greedy(home, markets, need):
    """Return (route as a list of names, total distance including the leg back home)."""
    check_feasible(markets, need)
    pos, left = home, set(need)
    route, total = [], 0.0
    while left:
        # only consider places that still sell a missing ingredient; on a tie take the earlier place
        d, i = min(
            (dist(pos, m["pos"]), i)
            for i, m in enumerate(markets)
            if left & m["items"]
        )
        total += d
        pos = markets[i]["pos"]
        route.append(markets[i]["name"])
        left -= markets[i]["items"]  # buy everything this place has
    total += dist(pos, home)  # leg back home
    return route, total


def heuristic(pos, left, home, markets):
    """
    Admissible and consistent heuristic.
    For each missing ingredient: pick the seller that gives the shortest trip pos -> seller -> home.
    Take the max over ingredients (no summing, to avoid counting the same trip twice).
    """
    if not left:
        return dist(pos, home)
    return max(
        min(
            dist(pos, m["pos"]) + dist(m["pos"], home)
            for m in markets
            if item in m["items"]
        )
        for item in left
    )


def astar(home, markets, need):
    """
    Return (route, total distance, number of expanded nodes).
    State: (position index, frozenset of missing ingredients); HOME = -1.
    Goal: everything bought and back at home.
    """
    check_feasible(markets, need)
    need = frozenset(need)
    tie = itertools.count()  # avoids comparing lists/sets when f values are equal

    def position(loc):
        return home if loc == HOME else markets[loc]["pos"]

    start = (HOME, need)
    # heap entry: (f, tie-breaker, g, position, missing ingredients, path)
    open_heap = [(heuristic(home, need, home, markets), next(tie), 0.0, HOME, need, ())]
    best_g = {}
    expanded = 0

    while open_heap:
        f, _, g, loc, left, path = heapq.heappop(open_heap)

        # skip if this state was already reached with a better or equal g
        if best_g.get((loc, left), math.inf) <= g:
            continue
        best_g[(loc, left)] = g
        expanded += 1

        # goal test: everything bought and back home
        if not left and loc == HOME and path:
            return [markets[i]["name"] for i in path], g, expanded

        pos = position(loc)

        if not left:
            # everything bought but not home yet: only the leg home remains
            ng = g + dist(pos, home)
            heapq.heappush(open_heap, (ng, next(tie), ng, HOME, left, path))
            continue

        # still missing items: try each place that sells at least one missing ingredient
        for i, m in enumerate(markets):
            if i == loc or not (left & m["items"]):
                continue
            new_left = left - m["items"]
            ng = g + dist(pos, m["pos"])
            nf = ng + heuristic(m["pos"], new_left, home, markets)
            heapq.heappush(open_heap, (nf, next(tie), ng, i, new_left, path + (i,)))

    return None, math.inf, expanded  # no solution


# ---------------------------------------------------------------- performance measurement
def measure(func, *args, repeat=20):
    """
    Run func(*args) and return (result, average time (seconds), peak memory (bytes)).

    - Time: average of `repeat` runs, measured with time.perf_counter().
      tracemalloc is NOT enabled here so the measurement is not slowed down.
    - Memory: Python allocation peak (tracemalloc) in ONE separate run. Counts only the memory
      the algorithm allocates while running (open list, g table, paths...), not the input data
      that was already loaded beforehand.
    """
    result = None
    total_time = 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(*args)
        total_time += time.perf_counter() - t0

    tracemalloc.start()
    func(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, total_time / repeat, peak


def count_steps(route):
    """Steps = number of moves = places visited + 1 (the last move is back home)."""
    return len(route) + 1


# ---------------------------------------------------------------- run and report
def fmt_route(route):
    return " -> ".join(["Home", *route, "Home"])


def run(path, key, repeat=20):
    """
    Run Greedy and A* on one input, print a comparison table with 3 groups of criteria and return a result dict.
    """
    home, markets, need = load_input(path, key)

    (g_route, g_cost), g_time, g_mem = measure(greedy, home, markets, need, repeat=repeat)
    (a_route, a_cost, a_nodes), a_time, a_mem = measure(astar, home, markets, need, repeat=repeat)

    # A* with an admissible, consistent h gives the true shortest route -> used as the benchmark
    opt_cost = a_cost
    opt_steps = count_steps(a_route)

    def metrics(route, cost, t, mem):
        steps = count_steps(route)
        return {
            "time_s": t,                          # average execution time (seconds)
            "peak_bytes": mem,                    # peak memory (bytes)
            "steps": steps,                       # steps (number of moves)
            "steps_vs_opt": steps - opt_steps,    # step difference vs. the shortest route
            "cost_km": cost,                      # distance (km)
            "optimality_pct": opt_cost / cost * 100 if cost else 100.0,
            "route": route,
        }

    g = metrics(g_route, g_cost, g_time, g_mem)
    a = metrics(a_route, a_cost, a_time, a_mem)

    line = "-" * 64
    print(f"=== {key}: {len(markets)} places, {len(need)} ingredients "
          f"(time = average of {repeat} runs) ===")
    print(f"{'Criterion':<40}{'Greedy':>12}{'A*':>12}")
    print(line)
    print(f"{'Average execution time (s)':<40}{g['time_s']:>12.6f}{a['time_s']:>12.6f}")
    print(f"{'Peak memory usage (bytes)':<40}{g['peak_bytes']:>12,}{a['peak_bytes']:>12,}")
    print(f"{'Steps (moves, incl. return home)':<40}{g['steps']:>12}{a['steps']:>12}")
    print(f"{'Steps vs. shortest path':<40}{g['steps_vs_opt']:>+12}{a['steps_vs_opt']:>+12}")
    print(f"{'Distance (km)':<40}{g['cost_km']:>12.2f}{a['cost_km']:>12.2f}")
    print(f"{'Optimality (% = optimal / found)':<40}{g['optimality_pct']:>11.1f}%{a['optimality_pct']:>11.1f}%")
    print(f"{'A* nodes expanded':<40}{'-':>12}{a_nodes:>12,}")
    print(line)
    print(f"Greedy: {fmt_route(g_route)}")
    print(f"A*    : {fmt_route(a_route)}")
    print()

    a["expanded_nodes"] = a_nodes
    return {"key": key, "markets": len(markets), "items": len(need), "greedy": g, "astar": a}


def print_summary(results):
    """Compact summary table when running several inputs."""
    print("=== SUMMARY ===")
    head = (f"{'Input':<8}{'Avg time (s)':>24}{'Peak memory (bytes)':>28}"
            f"{'Steps':>12}{'Optimality (%)':>20}")
    sub = (f"{'':<8}{'Greedy':>12}{'A*':>12}{'Greedy':>14}{'A*':>14}"
           f"{'G':>6}{'A*':>6}{'Greedy':>10}{'A*':>10}")
    print(head)
    print(sub)
    print("-" * len(sub))
    for r in results:
        g, a = r["greedy"], r["astar"]
        print(f"{r['key']:<8}{g['time_s']:>12.6f}{a['time_s']:>12.6f}"
              f"{g['peak_bytes']:>14,}{a['peak_bytes']:>14,}"
              f"{g['steps']:>6}{a['steps']:>6}"
              f"{g['optimality_pct']:>9.1f}%{a['optimality_pct']:>9.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Greedy and A* (return home after shopping)")
    parser.add_argument("file", nargs="?", default="markets_input.json")
    parser.add_argument("keys", nargs="*", help="e.g. input1 input2 (default: all)")
    parser.add_argument("--repeat", type=int, default=20, help="number of repeats used to average the execution time")
    args = parser.parse_args()

    keys = args.keys
    if not keys:
        with open(args.file, encoding="utf-8") as f:
            keys = [k for k in json.load(f) if k.startswith("input")]

    results = [run(args.file, k, repeat=args.repeat) for k in keys]
    if len(results) > 1:
        print_summary(results)
