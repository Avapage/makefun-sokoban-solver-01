// Sokoban solver: push-optimal A* over room sub-problems.
//
// Levels built as a hallway with small rooms off it decompose cleanly: a box
// can never travel from one room to another (leaving through a door strands
// it in the hallway for good), so a level's push optimum is the sum of its
// rooms' optima. This solver finds the rooms, solves each one push-optimally
// with an A* over pushes (player reachability folded into each state, a
// minimum-cost box-to-goal matching on true push distances as the heuristic,
// dead-square and frozen-block pruning), then stitches the room solutions
// together with plain walks along the hallway. If a level does not decompose
// the way the generator builds them, it falls back to the same search over
// the whole board. All budgets are counted in expanded states, never in
// time, so two runs of the program write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Direction tables: 0 up, 1 down, 2 left, 3 right. Opposite of d is d ^ 1.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const INF_COST = 1000000;
// Work budgets, counted in expanded states so runs stay deterministic: one
// search attempt gives up after ATTEMPT_STATES expansions, and the whole
// set of instances shares TOTAL_STATES between them.
const ATTEMPT_STATES = 1500000;
const TOTAL_STATES = 24000000;

// ---------------------------------------------------------------------------
// Level parsing (XSB), mirroring the checker's reading of level.txt.
// ---------------------------------------------------------------------------
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  // Neighbour table: nbr[cell * 4 + dir] = the adjacent cell, or -1.
  const nbr = new Int32Array(n * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) nbr[i * 4 + d] = ny * w + nx;
      }
    }
  }
  const goals = [];
  for (let i = 0; i < n; i++) if (goal[i]) goals.push(i);
  return { w, h, n, wall, goal, goals, boxes, player, nbr };
}

// ---------------------------------------------------------------------------
// Push distances: distTable(board, goalCell, boxArea)[cell] is the fewest
// pushes that move a box from cell onto goalCell, other boxes ignored and the
// player assumed able to stand on any floor square. Reverse breadth-first
// search over box positions: a box now on x could have been pushed there from
// y = x - d, with the player standing on y - d.
// ---------------------------------------------------------------------------
function distTable(board, goalCell, boxArea) {
  const { n, wall, nbr } = board;
  const dist = new Int32Array(n).fill(-1);
  dist[goalCell] = 0;
  const queue = [goalCell];
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let d = 0; d < 4; d++) {
      const y = nbr[x * 4 + (d ^ 1)];
      if (y < 0 || !boxArea[y] || dist[y] >= 0) continue;
      const stand = nbr[y * 4 + (d ^ 1)];
      if (stand < 0 || wall[stand]) continue;
      dist[y] = dist[x] + 1;
      queue.push(y);
    }
  }
  return dist;
}

// Minimum-cost assignment of boxes to goals (Hungarian method, O(n^3)).
// Costs at or above INF_COST count as forbidden; an assignment forced onto a
// forbidden pair reports Infinity, which the search reads as a dead state.
function assignmentCost(cost) {
  const n = cost.length;
  if (n === 0) return 0;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    if (p[j] > 0) total += cost[p[j] - 1][j - 1];
  }
  return total >= INF_COST ? Infinity : total;
}

// ---------------------------------------------------------------------------
// One push-search problem: a board, the squares a box may occupy, the squares
// the player may walk on, goals, the starting boxes and the player's start.
// ---------------------------------------------------------------------------
function makeProblem(board, boxArea, walkArea, goals, startBoxes, startPlayer) {
  const tables = goals.map((g) => distTable(board, g, boxArea));
  const nearest = new Int32Array(board.n).fill(-1);
  for (let i = 0; i < board.n; i++) {
    let best = -1;
    for (const t of tables) {
      if (t[i] >= 0 && (best < 0 || t[i] < best)) best = t[i];
    }
    nearest[i] = best;
  }
  return {
    board,
    boxArea,
    walkArea,
    goals,
    tables,
    nearest,
    startBoxes,
    startPlayer,
    goalSet: board.goal,
  };
}

// Player flood fill: fills `out` with the squares the player reaches from
// `from` without crossing a wall, a box or the edge of the walk area, and
// returns the smallest reached square (the state's canonical player square).
function flood(prob, from, occ, out) {
  const { board, walkArea } = prob;
  out.fill(0);
  let min = from;
  const stack = [from];
  out[from] = 1;
  while (stack.length > 0) {
    const c = stack.pop();
    if (c < min) min = c;
    for (let d = 0; d < 4; d++) {
      const q = board.nbr[c * 4 + d];
      if (q >= 0 && walkArea[q] && !occ[q] && !out[q]) {
        out[q] = 1;
        stack.push(q);
      }
    }
  }
  return min;
}

// A 2x2 block of walls and boxes never gives up any of its boxes: if one of
// them sits off a goal the position is dead. Checks the blocks covering cell.
function frozenBlock(board, occ, cell) {
  const { wall, goal, nbr } = board;
  const solid = (c) => c < 0 || wall[c] === 1 || occ[c] === 1;
  const dirs = [
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
  ];
  for (const [dv, dh] of dirs) {
    const v = nbr[cell * 4 + dv];
    const hz = nbr[cell * 4 + dh];
    const diag = v < 0 ? -1 : nbr[v * 4 + dh];
    if (solid(cell) && solid(v) && solid(hz) && solid(diag)) {
      const cells = [cell, v, hz, diag];
      let stuckBoxOffGoal = false;
      for (const c of cells) {
        if (c >= 0 && occ[c] && !goal[c]) stuckBoxOffGoal = true;
      }
      if (stuckBoxOffGoal) return true;
    }
  }
  return false;
}

// A box wedged in a wall corner, off its goal, can never move again.
function wallCorner(board, occ, cell) {
  const { wall, nbr } = board;
  const blocked = (d) => {
    const c = nbr[cell * 4 + d];
    return c < 0 || wall[c] === 1;
  };
  return (blocked(0) || blocked(1)) && (blocked(2) || blocked(3));
}

// The push search itself. mode 'astar' is exact (f = g + h); 'weighted'
// inflates the heuristic to reach a goal sooner; 'greedy' follows the
// heuristic alone. Returns { moves, pushes, endPlayer } or null when the
// state budget runs out first. Deterministic: fixed direction order, a heap
// ordered by (priority, depth, insertion order), budgets counted in states.
function search(prob, mode, budget) {
  const { board, boxArea, goals, tables } = prob;
  const n = board.n;
  const weight = mode === 'weighted' ? 3 : 1;
  const occ = new Uint8Array(n);
  const region = new Uint8Array(n);
  const childRegion = new Uint8Array(n);
  const hCache = new Map();

  const heuristic = (boxes) => {
    const key = String.fromCharCode(...boxes);
    let h = hCache.get(key);
    if (h !== undefined) return h;
    const cost = boxes.map((b) => tables.map((t) => (t[b] < 0 ? INF_COST : t[b])));
    h = assignmentCost(cost);
    hCache.set(key, h);
    return h;
  };

  // Node store, struct-of-arrays style.
  const nBoxes = [];
  const nPlayer = [];
  const nG = [];
  const nPrio = [];
  const nParent = [];
  const nDir = [];
  const nFrom = [];
  const nTo = [];
  const best = new Map();
  const heap = [];
  const before = (a, b) => {
    if (nPrio[a] !== nPrio[b]) return nPrio[a] < nPrio[b];
    if (nG[a] !== nG[b]) return nG[a] > nG[b];
    return a < b;
  };
  const heapPush = (id) => {
    heap.push(id);
    let i = heap.length - 1;
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (!before(heap[i], heap[par])) break;
      [heap[i], heap[par]] = [heap[par], heap[i]];
      i = par;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && before(heap[l], heap[m])) m = l;
        if (r < heap.length && before(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]];
        i = m;
      }
    }
    return top;
  };

  const addNode = (boxes, player, canon, g, parent, dir, from, to) => {
    const h = heuristic(boxes);
    if (h === Infinity) return -1;
    const key = String.fromCharCode(canon, ...boxes);
    const seen = best.get(key);
    if (seen !== undefined && seen <= g) return -1;
    best.set(key, g);
    const id = nBoxes.length;
    nBoxes.push(boxes);
    nPlayer.push(player);
    nG.push(g);
    nPrio.push(mode === 'greedy' ? h : g + weight * h);
    nParent.push(parent);
    nDir.push(dir);
    nFrom.push(from);
    nTo.push(to);
    heapPush(id);
    return id;
  };

  const startBoxes = [...prob.startBoxes].sort((a, b) => a - b);
  occ.fill(0);
  for (const b of startBoxes) occ[b] = 1;
  const startCanon = flood(prob, prob.startPlayer, occ, region);
  addNode(startBoxes, prob.startPlayer, startCanon, 0, -1, -1, -1, -1);

  let goalId = -1;
  let spent = 0;
  while (heap.length > 0) {
    if (spent >= ATTEMPT_STATES || budget.left <= 0) return null;
    const id = heapPop();
    const boxes = nBoxes[id];
    const g = nG[id];
    occ.fill(0);
    for (const b of boxes) occ[b] = 1;
    const canon = flood(prob, nPlayer[id], occ, region);
    const key = String.fromCharCode(canon, ...boxes);
    if (best.get(key) < g) continue;
    spent++;
    budget.left--;
    if (boxes.every((b) => prob.goalSet[b])) {
      goalId = id;
      break;
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const stand = board.nbr[b * 4 + (d ^ 1)];
        if (stand < 0 || !region[stand]) continue;
        const tgt = board.nbr[b * 4 + d];
        if (tgt < 0 || board.wall[tgt] || !boxArea[tgt] || occ[tgt]) continue;
        if (!prob.goalSet[tgt]) {
          if (prob.nearest[tgt] < 0) continue;
          if (wallCorner(board, occ, tgt)) continue;
        }
        const childBoxes = boxes.slice();
        childBoxes[bi] = tgt;
        childBoxes.sort((a, b2) => a - b2);
        // Judge the child in its own configuration: the pushed box's old
        // square is empty now (the player stands there), so it must not
        // count as a blockage in the frozen-block test or the flood.
        occ[b] = 0;
        occ[tgt] = 1;
        const frozen = !prob.goalSet[tgt] && frozenBlock(board, occ, tgt);
        const childCanon = frozen ? -1 : flood(prob, b, occ, childRegion);
        occ[b] = 1;
        occ[tgt] = 0;
        if (frozen) continue;
        addNode(childBoxes, b, childCanon, g + 1, id, d, b, tgt);
      }
    }
  }
  if (goalId < 0) return null;

  // Rebuild the push list, then spell out the walks between pushes.
  const pushes = [];
  for (let id = goalId; nParent[id] >= 0; id = nParent[id]) {
    pushes.push({ dir: nDir[id], from: nFrom[id], to: nTo[id] });
  }
  pushes.reverse();
  const simOcc = new Uint8Array(n);
  for (const b of startBoxes) simOcc[b] = 1;
  let player = prob.startPlayer;
  let moves = '';
  const scratch = new Uint8Array(n);
  for (const p of pushes) {
    const stand = board.nbr[p.from * 4 + (p.dir ^ 1)];
    moves += walkPath(prob, player, stand, simOcc, scratch);
    moves += LETTER[p.dir];
    simOcc[p.from] = 0;
    simOcc[p.to] = 1;
    player = p.from;
  }
  return { moves, pushes: pushes.length, endPlayer: player };
}

// Shortest walk of the player from `from` to `to` around the boxes in occ,
// as move letters; the squares the player may use are the problem's walk
// area. Returns '' when already there (or, defensively, when unreachable).
function walkPath(prob, from, to, occ, scratch) {
  if (from === to) return '';
  const { board, walkArea } = prob;
  const prev = new Int32Array(board.n).fill(-2);
  prev[from] = -1;
  const queue = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    if (c === to) break;
    for (let d = 0; d < 4; d++) {
      const q = board.nbr[c * 4 + d];
      if (q >= 0 && walkArea[q] && !occ[q] && prev[q] === -2) {
        prev[q] = c;
        queue.push(q);
      }
    }
  }
  if (prev[to] === -2) return '';
  const letters = [];
  let c = to;
  while (c !== from) {
    const p = prev[c];
    for (let d = 0; d < 4; d++) {
      if (board.nbr[p * 4 + d] === c) letters.push(LETTER[d]);
    }
    c = p;
  }
  letters.reverse();
  return letters.join('');
}

// Solve one problem exactly; if the exact search exhausts its budget, fall
// back to faster, possibly push-heavier searches so the level still solves.
function solveProblem(prob, budget) {
  const exact = search(prob, 'astar', budget);
  if (exact !== null) return exact;
  const weighted = search(prob, 'weighted', budget);
  if (weighted !== null) return weighted;
  return search(prob, 'greedy', budget);
}

// ---------------------------------------------------------------------------
// Room decomposition. The player's row is the hallway. The squares directly
// above and below it that are not wall are doors; cutting the hallway and the
// doors out of the floor leaves each room's inside as one connected piece.
// ---------------------------------------------------------------------------
function decompose(board, boxes, player) {
  const { w, h, n, wall, nbr } = board;
  const hy = Math.floor(player / w);
  for (let x = 1; x < w - 1; x++) {
    if (wall[hy * w + x]) return null;
  }
  const hallway = new Uint8Array(n);
  for (let x = 1; x < w - 1; x++) hallway[hy * w + x] = 1;
  const doorSet = new Uint8Array(n);
  const doors = [];
  for (let x = 1; x < w - 1; x++) {
    for (const dy of [-1, 1]) {
      const y = hy + dy;
      if (y < 0 || y >= h) continue;
      const c = y * w + x;
      if (!wall[c] && !doorSet[c]) {
        doorSet[c] = 1;
        doors.push(c);
      }
    }
  }
  if (doors.length === 0) return null;
  // Connected pieces of the floor that is neither hallway nor door.
  const comp = new Int32Array(n).fill(-1);
  const pieces = [];
  for (let i = 0; i < n; i++) {
    if (wall[i] || hallway[i] || doorSet[i] || comp[i] >= 0) continue;
    const cells = [];
    const stack = [i];
    comp[i] = pieces.length;
    while (stack.length > 0) {
      const c = stack.pop();
      cells.push(c);
      for (let d = 0; d < 4; d++) {
        const q = nbr[c * 4 + d];
        if (q >= 0 && !wall[q] && !hallway[q] && !doorSet[q] && comp[q] < 0) {
          comp[q] = pieces.length;
          stack.push(q);
        }
      }
    }
    pieces.push(cells);
  }
  const boxAt = new Map();
  boxes.forEach((b, i) => boxAt.set(b, i));
  const rooms = [];
  const claimedBoxes = new Set();
  for (let pi = 0; pi < pieces.length; pi++) {
    const cells = pieces[pi];
    const cellSet = new Uint8Array(n);
    for (const c of cells) cellSet[c] = 1;
    const myDoors = doors.filter((dc) => {
      for (let d = 0; d < 4; d++) if (nbr[dc * 4 + d] >= 0 && comp[nbr[dc * 4 + d]] === pi) return true;
      return false;
    });
    if (myDoors.length !== 1) return null;
    const door = myDoors[0];
    const goals = cells.filter((c) => board.goal[c]);
    const roomBoxes = cells.filter((c) => boxAt.has(c));
    if (boxAt.has(door)) roomBoxes.push(door);
    if (goals.length === 0 || goals.length !== roomBoxes.length) return null;
    for (const b of roomBoxes) claimedBoxes.add(b);
    // The hallway square outside the door: the door's neighbour in the hall.
    let outside = -1;
    for (let d = 0; d < 4; d++) {
      const q = nbr[door * 4 + d];
      if (q >= 0 && hallway[q]) outside = q;
    }
    if (outside < 0) return null;
    const boxArea = new Uint8Array(n);
    const walkArea = new Uint8Array(n);
    for (const c of cells) {
      boxArea[c] = 1;
      walkArea[c] = 1;
    }
    boxArea[door] = 1;
    walkArea[door] = 1;
    for (let i = 0; i < n; i++) if (hallway[i]) walkArea[i] = 1;
    rooms.push({ boxArea, walkArea, goals, boxes: roomBoxes.sort((a, b) => a - b), outside });
  }
  if (claimedBoxes.size !== boxes.length) return null;
  rooms.sort((a, b) => a.outside - b.outside);
  return rooms;
}

// Replay letters on a scratch state, checking every step the way the
// checker does. Mutates occ and returns the player's final square, or -1.
function applyMoves(board, occ, player, moves) {
  let p = player;
  for (const ch of moves) {
    const d = LETTER.indexOf(ch);
    const next = board.nbr[p * 4 + d];
    if (next < 0 || board.wall[next]) return -1;
    if (occ[next]) {
      const beyond = board.nbr[next * 4 + d];
      if (beyond < 0 || board.wall[beyond] || occ[beyond]) return -1;
      occ[next] = 0;
      occ[beyond] = 1;
    }
    p = next;
  }
  return p;
}

// A walk on the whole board (used between rooms): every floor square is
// walkable, boxes are obstacles.
function boardWalk(board, from, to, occ) {
  const all = new Uint8Array(board.n).fill(1);
  for (let i = 0; i < board.n; i++) if (board.wall[i]) all[i] = 0;
  const prob = { board, walkArea: all };
  return walkPath(prob, from, to, occ, new Uint8Array(board.n));
}

function solveLevel(text, budget) {
  const board = parseLevel(text);
  const { boxes, player } = board;
  if (boxes.length === 0 || player < 0) return '';
  if (boxes.length !== board.goals.length) return '';
  const rooms = decompose(board, boxes, player);
  if (rooms !== null) {
    const occ = new Uint8Array(board.n);
    for (const b of boxes) occ[b] = 1;
    let p = player;
    let out = '';
    let ok = true;
    for (const room of rooms) {
      const prob = makeProblem(board, room.boxArea, room.walkArea, room.goals, room.boxes, room.outside);
      const solved = solveProblem(prob, budget);
      if (solved === null) {
        ok = false;
        break;
      }
      const walk = boardWalk(board, p, room.outside, occ);
      const afterWalk = applyMoves(board, occ, p, walk);
      if (afterWalk < 0) {
        ok = false;
        break;
      }
      p = afterWalk;
      const afterRoom = applyMoves(board, occ, p, solved.moves);
      if (afterRoom < 0) {
        ok = false;
        break;
      }
      p = afterRoom;
      out += walk + solved.moves;
    }
    if (ok) {
      let allOn = true;
      for (let i = 0; i < board.n; i++) if (occ[i] && !board.goal[i]) allOn = false;
      if (allOn) return out;
    }
  }
  // Fallback: one search over the whole board.
  const all = new Uint8Array(board.n);
  for (let i = 0; i < board.n; i++) if (!board.wall[i]) all[i] = 1;
  const prob = makeProblem(board, all, all, board.goals, boxes, player);
  const solved = solveProblem(prob, budget);
  return solved === null ? '' : solved.moves;
}

function solve(text, budget) {
  try {
    return solveLevel(text, budget);
  } catch {
    return '';
  }
}

// One shared, work-counted budget for the whole set of instances.
const budget = { left: TOTAL_STATES };
for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'), budget);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
