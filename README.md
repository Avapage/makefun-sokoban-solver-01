# Sokoban Solver — Room-Decomposed Push-Optimal A*

[![Competition](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Avapage/makefun-sokoban-solver-01/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Avapage/makefun-sokoban-solver-01/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

Entry for the make.fun bounty [sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121): one program (`main.ts`, Node 22, no dependencies) that reads every Sokoban level in `/input/<id>/level.txt` and writes the player's moves to `/output/<id>/solution.txt`, minimizing pushes.

## How it works

Generated levels are a hallway with small rooms off it, and a box can never travel between rooms — a box pushed out of a door is stranded in the hallway for good. So a level's push optimum is exactly the sum of its rooms' optima, and the solver exploits that:

1. **Decompose.** The player's row is the hallway; the non-wall squares directly above and below it are doors. Cutting the hallway and doors out of the floor graph leaves each room's inside as one connected piece, with its own goals, boxes (one may start on the door square) and the hallway square outside its door.
2. **Solve each room push-optimally** with an A* over pushes:
   - states are (sorted box squares, canonical player square) — the player is represented by the smallest square of its reachable region, so walks never multiply states;
   - the heuristic is a minimum-cost box→goal assignment (Hungarian method) over true push distances, precomputed per goal by reverse breadth-first search over box positions — admissible, so the first goal reached is push-optimal;
   - dead squares (no push path to any goal), wall corners and frozen 2×2 blocks prune children before they are queued.
3. **Stitch.** Room solutions are replayed in door order, joined by shortest player walks along the hallway, and every step is re-checked against the full board as it is appended.

If a level ever fails to decompose, the same search runs over the whole board instead. If the exact search exhausts its budget, weighted and greedy variants still return a legal solving move string rather than nothing.

All budgets are counted in expanded states, never in wall-clock time, so two executions of the program write byte-identical answers. On the kit's practice set the solver finishes every instance at the known optimum (13/16/29/47/60 pushes) in well under a second for the whole set.
