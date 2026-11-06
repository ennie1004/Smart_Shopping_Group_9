# Shopping route: Greedy vs A*

Find the shortest route that buys every ingredient on a shopping list, then returns home.
Each market/supermarket sells some of the ingredients. The project compares a **Greedy** search
(always go to the nearest useful place) with **A\*** (optimal, with an admissible heuristic).

## Presentation

- Slides (Canva): https://canva.link/a3c56t7r2p2z4se

## Problem model

- Start at **Home** `(0, 0)`, buy everything, and **return home**.
- Every place has coordinates `(x, y)` in km. Distance is Euclidean: `d(a, b) = sqrt((x1-x2)^2 + (y1-y2)^2)`.
- Visiting a place buys **every missing ingredient it sells**, so revisiting a place never helps.
- Every place can go straight home (complete graph, triangle inequality holds).

## Files

| File | What it is |
|---|---|
| `markets_input.json` | The 5 inputs, plus reference results for checking your own code |
| `greedy_astar.py` | Greedy and A* implementations; `run()` prints time, peak memory and optimality |
| `visualize.py` | matplotlib: map of routes, A* step by step, Greedy step by step |
| `trace_astar_input4.gif` | Animation of A* running step by step on input 4 |
| `trace_greedy_input4.gif` | Animation of Greedy running step by step on input 4 |
| `Group 9 - Topic 5_ Smart Shopping - Report.pdf` | Written report |
| `.gitignore` | Files Git should ignore (`__pycache__/`, `out_dir/`, `*.pyc`) |

Keep `greedy_astar.py`, `visualize.py` and `markets_input.json` in the same folder.

## Requirements

- Python 3.8+.
- `greedy_astar.py` and `greedy_improve.py` use only the standard library.
- `visualize.py` needs `matplotlib`.

## The inputs

| Input | Places | Ingredients | Purpose |
|---|---|---|---|
| input1 | 5 | 8 | Small base case, Greedy is optimal |
| input2 | 7 | 12 | Greedy visits a nearby but redundant place |
| input3 | 10 | 15 | Greedy is lured by a nearby market and by a far supermarket |
| input4 | 12 | 18 | Greedy is optimal; built as a trap for a non-admissible heuristic |
| input5 | 15 | 24 | Greedy is far from optimal; built as a trap for a non-admissible heuristic |

`markets_input.json` structure, one entry per input:

```
inputN
  start          home name and position
  shopping_list  ingredients to buy
  markets        [{name, pos, items}, ...]
  reference      optimal / greedy / astar_h_admissible / astar_h_sum (routes and distances in km)
```

Reference distances (km), with return home:

| Input | Optimal = A* | Greedy | Greedy longer by |
|---|---|---|---|
| 1 | 16.37 | 16.37 | 0% |
| 2 | 20.92 | 22.29 | 6.5% |
| 3 | 23.86 | 26.09 | 9.3% |
| 4 | 17.63 | 17.63 | 0% |
| 5 | 20.46 | 55.75 | about 172% |

The optimal values were confirmed with a brute-force search / Held-Karp dynamic programming.
`astar_h_sum` in the JSON is a *non-admissible* variant (explained below) kept only to show how a bad
heuristic makes A* return a worse route. It is not used by any script.

## Algorithms

**Greedy.** From the current position, go to the nearest place that still sells at least one missing
ingredient (ties: the place listed first). Repeat until nothing is missing, then go home.

**A\*.** A state is `(position p, set S of missing ingredients)`. `g` is the distance travelled so far,
`f = g + h`, and the goal is "nothing missing and back home".

Heuristic (admissible and consistent):

```
h(p, S) = d(p, Home)                                         if S is empty
h(p, S) = max_{i in S}  min_{m sells i}  [ d(p, m) + d(m, Home) ]    otherwise
```

Reading it from the inside out:

1. `d(p, m) + d(m, Home)` is one trip: go from `p` to place `m`, then home.
2. `min` over the places `m` that sell ingredient `i` gives the cheapest way to get `i` and go home.
3. `max` over the missing ingredients picks the one that forces the longest trip.

It never overestimates the true remaining cost: the real route must still visit a seller of every
missing ingredient and then go home, and by the triangle inequality that cannot be shorter than the
trip counted above. It uses `max`, not a sum, because one visit can buy many ingredients and a sum would
count the same trip several times. A heuristic that sums the per-ingredient costs is not admissible,
which is what inputs 4 and 5 are designed to expose.

## Usage

### Compare Greedy and A* (`greedy_astar.py`)

```
python greedy_astar.py                                # all inputs, 20 repeats for timing
python greedy_astar.py markets_input.json input2      # one input
python greedy_astar.py markets_input.json --repeat 50 # change the number of repeats
```

For each input it prints, for Greedy and A*:

- average execution time (seconds),
- peak memory usage (bytes, Python allocations measured with `tracemalloc`),
- route optimality: number of steps (moves including the way home), steps vs the shortest route,
  distance (km) and optimality (% = optimal / found).

It also prints the number of A* nodes expanded and, when several inputs run, a summary table.
`run(path, key, repeat)` returns a dict with the same numbers if you want to plot or reuse them.
Times depend on your machine; the ratios between the two algorithms are what matter.

### Improve Greedy (`greedy_improve.py`)

```
python greedy_improve.py                         # all inputs
python greedy_improve.py markets_input.json input5
```

Shows the distance and the gap to the optimal route (A*) after each step: plain Greedy, remove
redundant stops, 2-opt, and multi-start (try every place as the first stop, then improve). Output is in
English.

### Visualize (`visualize.py`)

```
# map (Greedy and A* routes together)
python visualize.py map                                   # interactive window, pick the input with buttons
python visualize.py map --save out_dir                    # one PNG per input
python visualize.py map --input input3                    # a single input

# trace of the A* algorithm only
python visualize.py trace-astar --input input4            # interactive window, slider over steps
python visualize.py trace-astar --input input4 --save out_dir

# trace of the Greedy algorithm only
python visualize.py trace-greedy --input input4           # interactive window, slider over steps
python visualize.py trace-greedy --input input4 --save out_dir
```

- `--input` defaults to `input4` for both trace modes; `map` draws all inputs when it is omitted.
- `map` shows Home, the places (circle size = number of items sold), both routes, and on the right the
  ingredients to buy and what each place sells.
- `trace-astar` shows, per step, the node being expanded, its new children with `g`, `h`, `f`, and the
  OPEN list ordered by `f`.
- `trace-greedy` shows, per step, the candidate places with their distances, the nearest one chosen,
  what gets bought and what is still missing.
- Saved files: `map_<input>.png`, `trace_astar_<input>_NN.png`, `trace_greedy_<input>_NN.png`.

`visualize_5_inputs.html` is a standalone page: choose an input, toggle the Greedy and A* routes, and
click an ingredient to highlight the places that sell it.

## Scaling notes

The problem combines set cover (which places to visit) and TSP (in what order), so it is NP-hard.

- A* with this heuristic worked comfortably up to roughly 20-25 places on random test data
  (Python); some 30-place instances did not finish within a 150,000-node limit.
- Greedy is fast and light but its gap to optimal grows with size (input 5: 172%).
- Improving Greedy (`greedy_improve.py`) brings the gap down a lot at almost no cost
  (input 5: about 0.9%), but gives no guarantee.
- For much larger data, consider weighted A*, beam search, metaheuristics, or a MILP/OR-Tools solver,
  and keep A* as the reference on small instances.

These scaling figures come from random instances I generated for a quick experiment, not from a
benchmark suite, so treat them as a rough guide.
