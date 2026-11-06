"""
Three modes:
  map          : map of the places + the routes found by Greedy and by A* (all inputs, or one)
  trace-astar  : A* step by step (expand the node with the smallest f = g + h, g/h/f table, OPEN list)
  trace-greedy : Greedy step by step (candidate places and distances, the nearest one is chosen)

Usage:
    # map (Greedy and A* routes together), can replace input3 with any input from 1 to 5
    python visualize.py map                                   # interactive window, pick the input with buttons
    python visualize.py map --save out_dir                    # save one PNG per input
    python visualize.py map --input input3                    # a single input

    # trace of the A* algorithm only, can replace input4 with any input from 1 to 5
    python visualize.py trace-astar --input input4                   # interactive window, slider over A* steps
    python visualize.py trace-astar --input input4 --save out_dir    # save one PNG per A* step

    # trace of the Greedy algorithm only, can replace input4 with any input from 1 to 5
    python visualize.py trace-greedy --input input4                  # interactive window, slider over Greedy steps
    python visualize.py trace-greedy --input input4 --save out_dir   # save one PNG per Greedy step

--input defaults to input4 for trace-astar and trace-greedy (map: all inputs).
Saved files: map_<input>.png, trace_astar_<input>_NN.png, trace_greedy_<input>_NN.png
"""
import argparse
import heapq
import itertools
import json
import math
import os

import matplotlib

import greedy_astar as ga

COL = {"greedy": "#BA7517", "astar": "#1D9E75"}
STYLE = {"greedy": "--", "astar": "-"}
LABEL = {"greedy": "Greedy", "astar": "A* (h admissible)"}


# ------------------------------------------------------------------ data
def solve(path, key):
    home, markets, need = ga.load_input(path, key)
    index = {m["name"]: i for i, m in enumerate(markets)}
    g_route, g_cost = ga.greedy(home, markets, need)
    a_route, a_cost, _ = ga.astar(home, markets, need)
    with open(path, encoding="utf-8") as f:
        shopping = json.load(f)[key]["shopping_list"]  # keep the original order
    return {
        "home": home, "markets": markets, "need": shopping,
        "greedy": ([index[n] for n in g_route], g_cost),
        "astar": ([index[n] for n in a_route], a_cost),
    }


def radius(m):
    return 0.18 + 0.05 * math.sqrt(len(m["items"]))  # radius in km on the map


# ------------------------------------------------------------------ 1. map
def draw_map(ax, S, show=("greedy", "astar")):
    ax.clear()
    home, M = S["home"], S["markets"]
    for i, m in enumerate(M):
        ax.add_patch(matplotlib.patches.Circle(
            m["pos"], radius(m), fill=True, zorder=3, lw=1, fc="white", ec="#666"))
        ax.text(*m["pos"], str(i + 1), ha="center", va="center", fontsize=9, zorder=4)
    ax.add_patch(matplotlib.patches.Rectangle((home[0] - .2, home[1] - .2), .4, .4, fc="black", zorder=4))
    ax.text(home[0] + .35, home[1] + .3, "Home", fontsize=9, weight="bold")

    def pos(i):
        return home if i < 0 else M[i]["pos"]

    for k, off in (("greedy", -0.07), ("astar", 0.07)):
        if k not in show:
            continue
        seq = [-1, *S[k][0], -1]
        for a, b in zip(seq, seq[1:]):
            (x1, y1), (x2, y2) = pos(a), pos(b)
            L = math.hypot(x2 - x1, y2 - y1) or 1
            ux, uy = (x2 - x1) / L, (y2 - y1) / L
            nx, ny = -uy * off, ux * off
            r1 = (0.25 if a < 0 else radius(M[a])) + .03
            r2 = (0.25 if b < 0 else radius(M[b])) + .09
            ax.annotate("", xy=(x2 - ux * r2 + nx, y2 - uy * r2 + ny),
                        xytext=(x1 + ux * r1 + nx, y1 + uy * r1 + ny),
                        arrowprops=dict(arrowstyle="->", color=COL[k], lw=1.8, ls=STYLE[k]), zorder=2)
    xs = [home[0]] + [m["pos"][0] for m in M]
    ys = [home[1]] + [m["pos"][1] for m in M]
    ax.set_xlim(-1, max(xs) + 1.5)
    ax.set_ylim(-1, max(ys) + 1.5)
    ax.set_aspect("equal")
    ax.grid(alpha=.25)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    opt = S["astar"][1]
    lines = [f"{len(M)} places, {len(S['need'])} ingredients"]
    for k in ("greedy", "astar"):
        cost = S[k][1]
        gap = "= optimal" if cost - opt < 5e-3 else f"+{(cost / opt - 1) * 100:.1f}% vs optimal"
        lines.append(f"{LABEL[k]}: {cost:.2f} km ({len(S[k][0]) + 1} steps, {gap})")
    ax.set_title("\n".join(lines), fontsize=10, loc="left")


def side_text(fig, S):
    """List of ingredients to buy and, for every place, the ingredients it sells (right-hand column)."""
    import textwrap
    fig.texts.clear()
    need = textwrap.fill(", ".join(S["need"]), 62)
    lines = []
    for i, m in enumerate(S["markets"]):
        head = f"{i + 1}. {m['name']} {tuple(m['pos'])} - {len(m['items'])} items:"
        body = textwrap.fill(", ".join(sorted(m["items"], key=S["need"].index)), 62, initial_indent="     ", subsequent_indent="     ")
        lines.append(head + "\n" + body)
    fig.text(0.60, 0.95, f"Ingredients to buy ({len(S['need'])}):", weight="bold", fontsize=9, va="top")
    n = need.count("\n") + 1
    fig.text(0.60, 0.925, need, fontsize=7.5, va="top")
    y = 0.925 - 0.026 * n - 0.03
    fig.text(0.60, y, "Places and what each one sells:", weight="bold", fontsize=9, va="top")
    fig.text(0.60, y - 0.025, "\n".join(lines), fontsize=7, va="top", linespacing=1.25)


def cmd_map(args):
    import matplotlib.pyplot as plt
    keys = [args.input] if args.input else [k for k in json.load(open(args.file, encoding="utf-8")) if k.startswith("input")]
    data = {k: solve(args.file, k) for k in keys}

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for k, S in data.items():
            fig, ax = plt.subplots(figsize=(15, 8.5))
            fig.subplots_adjust(left=.06, right=.57, top=.86, bottom=.08)
            draw_map(ax, S)
            side_text(fig, S)
            out = os.path.join(args.save, f"map_{k}.png")
            fig.savefig(out, dpi=130)
            plt.close(fig)
            print("saved", out)
        return

    from matplotlib.widgets import CheckButtons, RadioButtons
    state = {"key": keys[0], "show": ["greedy", "astar"]}
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.subplots_adjust(left=.17, right=.57, top=.86, bottom=.08)

    def redraw():
        S = data[state["key"]]
        draw_map(ax, S, show=state["show"])
        side_text(fig, S)
        fig.canvas.draw_idle()

    rax = fig.add_axes([0.02, 0.55, 0.13, 0.30])
    radio = RadioButtons(rax, keys, active=0)
    radio.on_clicked(lambda lab: (state.update(key=lab), redraw()))
    cax = fig.add_axes([0.02, 0.38, 0.15, 0.12])
    chk = CheckButtons(cax, [LABEL["greedy"], LABEL["astar"]], [True, True])

    def toggle(_):
        st = chk.get_status()
        state["show"] = [k for k, on in zip(("greedy", "astar"), st) if on]
        redraw()
    chk.on_clicked(toggle)
    redraw()
    plt.show()


# ------------------------------------------------------------------ 2. trace-astar: A* step by step
def trace_astar(home, markets, need):
    """Re-run the exact loop of ga.astar and record every step."""
    need = frozenset(need)
    tie = itertools.count()
    nodes, steps = [], []

    def new_node(f, g, loc, left, path):
        nodes.append({"f": f, "g": g, "h": f - g, "loc": loc, "left": left, "path": path})
        return len(nodes) - 1

    f0 = ga.heuristic(home, need, home, markets)
    n0 = new_node(f0, 0.0, ga.HOME, need, ())
    heap = [(f0, next(tie), 0.0, ga.HOME, need, (), n0)]
    best = {}
    while heap:
        f, _, g, loc, left, path, nid = heapq.heappop(heap)
        if best.get((loc, left), math.inf) <= g:
            steps.append({"kind": "skip", "pop": nid, "push": [], "open": [x[6] for x in sorted(heap)]})
            continue
        best[(loc, left)] = g
        pushed = []
        if not left and loc == ga.HOME and path:
            steps.append({"kind": "goal", "pop": nid, "push": [], "open": [x[6] for x in sorted(heap)]})
            break
        pos = home if loc == ga.HOME else markets[loc]["pos"]
        if not left:
            ng = g + ga.dist(pos, home)
            n2 = new_node(ng, ng, ga.HOME, left, path)
            heapq.heappush(heap, (ng, next(tie), ng, ga.HOME, left, path, n2))
            pushed.append(n2)
        else:
            for i, m in enumerate(markets):
                if i == loc or not (left & m["items"]):
                    continue
                nl = left - m["items"]
                ng = g + ga.dist(pos, m["pos"])
                nf = ng + ga.heuristic(m["pos"], nl, home, markets)
                n2 = new_node(nf, ng, i, nl, path + (i,))
                heapq.heappush(heap, (nf, next(tie), ng, i, nl, path + (i,), n2))
                pushed.append(n2)
        steps.append({"kind": "exp", "pop": nid, "push": pushed, "open": [x[6] for x in sorted(heap)]})
    return nodes, steps


def draw_astar_step(fig, axm, axt, S, nodes, steps, k):
    """Draw step k of the A* trace: expanded node, new children (g, h, f) and the OPEN list."""
    home, M = S["home"], S["markets"]
    st = steps[k]
    cur = nodes[st["pop"]]
    axm.clear()
    axt.clear()
    axt.axis("off")

    def pos(i):
        return home if i < 0 else M[i]["pos"]

    # places already visited on the route being expanded
    visited = set(cur["path"])
    for i, m in enumerate(M):
        axm.add_patch(matplotlib.patches.Circle(
            m["pos"], radius(m), zorder=3, lw=1.5 if i in visited else 1,
            fc="#E1F5EE" if i in visited else "white", ec="#1D9E75" if i in visited else "#666"))
        axm.text(*m["pos"], str(i + 1), ha="center", va="center", fontsize=8, zorder=4)
    axm.add_patch(matplotlib.patches.Rectangle((home[0] - .2, home[1] - .2), .4, .4, fc="black", zorder=4))
    # children just generated (thin, orange)
    for nid in st["push"]:
        c = nodes[nid]
        if c["path"]:
            a = pos(c["path"][-2]) if len(c["path"]) > 1 else home
            b = pos(c["path"][-1])
            if c["loc"] != ga.HOME:
                axm.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1, alpha=.7), zorder=1)
    # route of the node being expanded (thick)
    seq = [-1, *cur["path"]] + ([-1] if cur["loc"] == ga.HOME and cur["path"] else [])
    for a, b in zip(seq, seq[1:]):
        axm.annotate("", xy=pos(b), xytext=pos(a), arrowprops=dict(arrowstyle="->", color="#1D9E75", lw=2.4), zorder=2)
    xs = [home[0]] + [m["pos"][0] for m in M]
    ys = [home[1]] + [m["pos"][1] for m in M]
    axm.set_xlim(-1, max(xs) + 1.5)
    axm.set_ylim(-1, max(ys) + 1.5)
    axm.set_aspect("equal")
    axm.grid(alpha=.25)

    def label(n):
        return " -> ".join(["Home", *[str(i + 1) for i in n["path"]]]) + (" -> Home" if n["loc"] == ga.HOME and n["path"] else "")

    title = {"exp": "Expand the node with the smallest f", "skip": "Skip (already reached with lower/equal g)",
             "goal": "GOAL: all items bought and back home"}[st["kind"]]
    axm.set_title(f"A* - step {k + 1}/{len(steps)}: {title}\n{label(cur)}   g={cur['g']:.2f}  h={cur['h']:.2f}  f={cur['f']:.2f}",
                  fontsize=10, loc="left")

    rows = ["New children (g, h, f):"]
    for nid in st["push"]:
        c = nodes[nid]
        rows.append(f"  {label(c):<30} g={c['g']:5.2f} h={c['h']:5.2f} f={c['f']:5.2f}  missing={len(c['left'])}")
    rows += ["", f"OPEN list ({len(st['open'])} nodes), smallest f first:"]
    for nid in st["open"][:8]:
        c = nodes[nid]
        rows.append(f"  {label(c):<30} f={c['f']:5.2f}")
    if len(st["open"]) > 8:
        rows.append(f"  ... {len(st['open']) - 8} more")
    axt.text(0, 1, "\n".join(rows), va="top", family="monospace", fontsize=7.5)


def cmd_trace_astar(args):
    """Mode trace-astar: record and show the A* run of one input (default input4)."""
    import matplotlib.pyplot as plt
    key = args.input or "input4"
    S = solve(args.file, key)
    nodes, steps = trace_astar(S["home"], S["markets"], frozenset(S["need"]))
    print(f"{key}: A* trace, {len(steps)} steps, {len(nodes)} nodes generated, "
          f"optimal = {nodes[steps[-1]['pop']]['f']:.2f} km")

    def make_fig():
        fig = plt.figure(figsize=(13, 6.8))
        axm = fig.add_axes([0.05, 0.1, 0.42, 0.78])
        axt = fig.add_axes([0.50, 0.1, 0.49, 0.78])
        return fig, axm, axt

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        files = []
        for k in range(len(steps)):
            fig, axm, axt = make_fig()
            draw_astar_step(fig, axm, axt, S, nodes, steps, k)
            out = os.path.join(args.save, f"trace_astar_{key}_{k + 1:02d}.png")
            fig.savefig(out, dpi=100)
            plt.close(fig)
            files.append(out)
        print(f"saved {len(files)} PNG in {args.save}")
        return

    from matplotlib.widgets import Slider
    fig, axm, axt = make_fig()
    fig.subplots_adjust(bottom=.12)
    sax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
    sl = Slider(sax, "Step", 1, len(steps), valinit=1, valstep=1)
    sl.on_changed(lambda v: (draw_astar_step(fig, axm, axt, S, nodes, steps, int(v) - 1), fig.canvas.draw_idle()))
    draw_astar_step(fig, axm, axt, S, nodes, steps, 0)
    plt.show()


# ------------------------------------------------------------------ 3. trace-greedy: Greedy step by step
def trace_greedy(home, markets, need):
    """Re-run the exact loop of ga.greedy and record every step (last step = the leg back home)."""
    ga.check_feasible(markets, need)
    pos, cur, left = home, ga.HOME, set(need)
    total = 0.0
    steps = []
    while left:
        cands = sorted(
            (ga.dist(pos, m["pos"]), i, len(left & m["items"]))
            for i, m in enumerate(markets) if left & m["items"]
        )
        d, i, _ = cands[0]  # nearest; on a tie the earlier place (same rule as ga.greedy)
        bought = left & markets[i]["items"]
        steps.append({"kind": "move", "from": cur, "cands": cands, "to": i, "d": d,
                      "bought": bought, "left_before": set(left), "total_before": total})
        total += d
        pos, cur = markets[i]["pos"], i
        left -= bought
    d = ga.dist(pos, home)
    steps.append({"kind": "home", "from": cur, "cands": [], "to": ga.HOME, "d": d,
                  "bought": set(), "left_before": set(), "total_before": total})
    total += d
    return steps, total


def draw_greedy_step(fig, axm, axt, S, steps, k):
    """Draw step k of the Greedy trace: candidates with distances, the nearest one is chosen."""
    home, M = S["home"], S["markets"]
    st = steps[k]
    axm.clear()
    axt.clear()
    axt.axis("off")

    def pos(i):
        return home if i < 0 else M[i]["pos"]

    visited = [s["to"] for s in steps[:k] if s["to"] >= 0]
    for i, m in enumerate(M):
        done = i in visited or i == st["to"] and st["kind"] == "move"
        axm.add_patch(matplotlib.patches.Circle(
            m["pos"], radius(m), zorder=3, lw=1.5 if done else 1,
            fc="#FAEEDA" if done else "white", ec=COL["greedy"] if done else "#666"))
        axm.text(*m["pos"], str(i + 1), ha="center", va="center", fontsize=8, zorder=4)
    axm.add_patch(matplotlib.patches.Rectangle((home[0] - .2, home[1] - .2), .4, .4, fc="black", zorder=4))

    # route already travelled (solid orange)
    seq = [-1, *visited]
    for a, b in zip(seq, seq[1:]):
        axm.annotate("", xy=pos(b), xytext=pos(a), arrowprops=dict(arrowstyle="->", color=COL["greedy"], lw=2, alpha=.6), zorder=2)
    # candidates considered at this step (thin grey) and the chosen move (thick)
    src = pos(st["from"])
    for d, i, _ in st["cands"]:
        axm.annotate("", xy=pos(i), xytext=src, arrowprops=dict(arrowstyle="-", color="#999", lw=.8, ls=":"), zorder=1)
        mx, my = (src[0] + pos(i)[0]) / 2, (src[1] + pos(i)[1]) / 2
        axm.text(mx, my, f"{d:.2f}", fontsize=7, color="#555", ha="center", zorder=5,
                 bbox=dict(fc="white", ec="none", pad=.5, alpha=.8))
    axm.annotate("", xy=pos(st["to"]), xytext=src, arrowprops=dict(arrowstyle="->", color=COL["greedy"], lw=2.8), zorder=2)
    axm.add_patch(matplotlib.patches.Circle(src, 0.28, fill=False, ec="#000", lw=1.2, ls="--", zorder=5))

    xs = [home[0]] + [m["pos"][0] for m in M]
    ys = [home[1]] + [m["pos"][1] for m in M]
    axm.set_xlim(-1, max(xs) + 1.5)
    axm.set_ylim(-1, max(ys) + 1.5)
    axm.set_aspect("equal")
    axm.grid(alpha=.25)

    def name(i):
        return "Home" if i < 0 else f"{i + 1}. {M[i]['name']}"

    order = lambda items: sorted(items, key=S["need"].index)
    if st["kind"] == "move":
        title = f"Greedy - step {k + 1}/{len(steps)}: go to the nearest place that sells a missing item"
        sub = f"{name(st['from'])} -> {name(st['to'])}   leg = {st['d']:.2f} km   total so far = {st['total_before'] + st['d']:.2f} km"
        rows = [f"Missing before this step ({len(st['left_before'])}):",
                "  " + ", ".join(order(st["left_before"])), "",
                "Candidates (places still selling a missing item), nearest first:",
                f"  {'place':<32}{'dist (km)':>10}{'new items':>11}"]
        for j, (d, i, n) in enumerate(st["cands"]):
            mark = "  <== chosen" if j == 0 else ""
            rows.append(f"  {name(i)[:31]:<32}{d:>10.2f}{n:>11}{mark}")
        rows += ["", f"Bought at {M[st['to']]['name']} ({len(st['bought'])}):",
                 "  " + ", ".join(order(st["bought"])),
                 "", f"Still missing after this step ({len(st['left_before'] - st['bought'])}):",
                 "  " + (", ".join(order(st["left_before"] - st["bought"])) or "nothing")]
    else:
        title = f"Greedy - step {k + 1}/{len(steps)}: everything bought, go back home"
        sub = f"{name(st['from'])} -> Home   leg = {st['d']:.2f} km   TOTAL = {st['total_before'] + st['d']:.2f} km"
        rows = ["All ingredients bought.", "",
                "Route: " + " -> ".join(["Home", *[str(i + 1) for i in visited], "Home"]),
                f"Distance: {st['total_before'] + st['d']:.2f} km",
                f"Steps: {len(steps)}"]
    axm.set_title(f"{title}\n{sub}", fontsize=9.5, loc="left")
    axt.text(0, 1, "\n".join(rows), va="top", family="monospace", fontsize=7.5)


def cmd_trace_greedy(args):
    """Mode trace-greedy: record and show the Greedy run of one input (default input4)."""
    import matplotlib.pyplot as plt
    key = args.input or "input4"
    S = solve(args.file, key)
    steps, total = trace_greedy(S["home"], S["markets"], frozenset(S["need"]))
    print(f"{key}: Greedy trace, {len(steps)} steps, total = {total:.2f} km")

    def make_fig():
        fig = plt.figure(figsize=(13, 6.8))
        axm = fig.add_axes([0.05, 0.1, 0.42, 0.78])
        axt = fig.add_axes([0.50, 0.1, 0.49, 0.78])
        return fig, axm, axt

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for k in range(len(steps)):
            fig, axm, axt = make_fig()
            draw_greedy_step(fig, axm, axt, S, steps, k)
            fig.savefig(os.path.join(args.save, f"trace_greedy_{key}_{k + 1:02d}.png"), dpi=100)
            plt.close(fig)
        print(f"saved {len(steps)} PNG in {args.save}")
        return

    from matplotlib.widgets import Slider
    fig, axm, axt = make_fig()
    fig.subplots_adjust(bottom=.12)
    sax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
    sl = Slider(sax, "Step", 1, len(steps), valinit=1, valstep=1)
    sl.on_changed(lambda v: (draw_greedy_step(fig, axm, axt, S, steps, int(v) - 1), fig.canvas.draw_idle()))
    draw_greedy_step(fig, axm, axt, S, steps, 0)
    plt.show()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="matplotlib visualizations of Greedy and A* (shop for all ingredients, then return home)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="modes:\n"
               "  map           map with the Greedy and A* routes\n"
               "  trace-astar   step-by-step trace of the A* algorithm\n"
               "  trace-greedy  step-by-step trace of the Greedy algorithm")
    p.add_argument("mode", choices=["map", "trace-astar", "trace-greedy"], help="what to draw (see modes below)")
    p.add_argument("file", nargs="?", default="markets_input.json")
    p.add_argument("--input", help="e.g. input4 (map: default all inputs; trace-astar / trace-greedy: default input4)")
    p.add_argument("--save", metavar="DIR", help="save PNGs instead of opening a window")
    a = p.parse_args()
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.patches  # noqa: F401
    {"map": cmd_map, "trace-astar": cmd_trace_astar, "trace-greedy": cmd_trace_greedy}[a.mode](a)
