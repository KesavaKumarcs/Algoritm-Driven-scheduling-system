
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        univ-scheduler — app.py                                 ║
║         Single-file: All DSA Backend (top) → Streamlit Frontend (bottom)       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Run: streamlit run app.py                                                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                     SECTION 1 — DSA BACKEND                               ██
# ██   GraphColoring · NQueens · MinHeap · PriorityQueue · TopologicalSort     ██
# ██                   BloomFilter · UnionFind · Backtracking                  ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

import math
import random
from collections import defaultdict, deque


# ══════════════════════════════════════════════════════════════════════════════
# 1.1  GRAPH COLORING
#      Welsh-Powell greedy + exact backtracking
#      Used for: Timetable conflict-free slot assignment
# ══════════════════════════════════════════════════════════════════════════════

class GraphColoring:
    """
    Courses = Nodes, Shared-student/faculty conflicts = Edges, Time slots = Colors.
    Greedy (Welsh-Powell) runs in O(V²+E).
    Backtracking finds the exact chromatic number but is O(k^V) worst-case.
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.adj = defaultdict(set)
        self.node_labels: dict[int, str] = {}
        self.steps: list[dict] = []

    # ── graph construction ──────────────────────────────────────────────────

    def add_edge(self, u: int, v: int):
        self.adj[u].add(v)
        self.adj[v].add(u)

    def set_label(self, node: int, label: str):
        self.node_labels[node] = label

    def degree(self, node: int) -> int:
        return len(self.adj[node])

    # ── algorithms ──────────────────────────────────────────────────────────

    def welsh_powell_order(self) -> list[int]:
        """Sort nodes by degree descending (higher degree = harder to color → schedule first)."""
        return sorted(range(self.num_nodes), key=lambda x: -self.degree(x))

    def greedy_color(self) -> dict[int, int]:
        """
        Greedy graph coloring.
        Returns {node_id: color_index}.  O(V² + E).
        Populates self.steps for step-by-step UI.
        """
        self.steps = []
        color_map: dict[int, int] = {}

        for node in self.welsh_powell_order():
            neighbor_colors = {color_map[nb] for nb in self.adj[node] if nb in color_map}
            color = 0
            while color in neighbor_colors:
                color += 1
            color_map[node] = color
            self.steps.append({
                'node': node,
                'color': color,
                'label': self.node_labels.get(node, f'N{node}'),
                'neighbors_colored': list(neighbor_colors),
                'snapshot': dict(color_map),
                'action': 'assign',
            })
        return color_map

    def backtrack_color(self, max_colors: int) -> dict[int, int] | None:
        """
        Exact backtracking coloring with at most max_colors.
        Returns color_map or None if infeasible.  O(max_colors^V) worst-case.
        """
        self.steps = []
        color_map: dict[int, int] = {}
        order = self.welsh_powell_order()

        def is_safe(node: int, color: int) -> bool:
            return all(color_map.get(nb) != color for nb in self.adj[node])

        def solve(idx: int) -> bool:
            if idx == len(order):
                return True
            node = order[idx]
            for color in range(max_colors):
                if is_safe(node, color):
                    color_map[node] = color
                    self.steps.append({'node': node, 'color': color,
                                       'label': self.node_labels.get(node, f'N{node}'),
                                       'action': 'assign', 'snapshot': dict(color_map)})
                    if solve(idx + 1):
                        return True
                    del color_map[node]
                    self.steps.append({'node': node, 'color': color,
                                       'label': self.node_labels.get(node, f'N{node}'),
                                       'action': 'backtrack', 'snapshot': dict(color_map)})
            return False

        return color_map if solve(0) else None

    def chromatic_number(self) -> int:
        """Binary-search + backtracking to find minimum colors needed."""
        upper = max(self.greedy_color().values(), default=0) + 1
        lo, hi, best = 1, upper, upper
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.backtrack_color(mid):
                best = mid;
                hi = mid - 1
            else:
                lo = mid + 1
        return best

    def bfs(self, start: int) -> list[int]:
        """BFS traversal from start node."""
        visited, queue, order = {start}, deque([start]), []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in sorted(self.adj[node]):
                if nb not in visited:
                    visited.add(nb);
                    queue.append(nb)
        return order

    def get_conflict_graph_data(self) -> tuple[list, list]:
        nodes = [{'id': i, 'label': self.node_labels.get(i, f'C{i}')} for i in range(self.num_nodes)]
        edges, seen = [], set()
        for u in range(self.num_nodes):
            for v in self.adj[u]:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    edges.append({'from': u, 'to': v});
                    seen.add(key)
        return nodes, edges

    @staticmethod
    def build_from_courses(courses: list[dict]) -> 'GraphColoring':
        """
        Build from list of dicts: {id, name, faculty, students}.
        Two courses conflict if they share a student or the same faculty.
        """
        n = len(courses)
        gc = GraphColoring(n)
        for i, c in enumerate(courses):
            gc.set_label(i, c['name'])
        for i in range(n):
            for j in range(i + 1, n):
                if (set(courses[i]['students']) & set(courses[j]['students'])
                        or (courses[i]['faculty'] and courses[i]['faculty'] == courses[j]['faculty'])):
                    gc.add_edge(i, j)
        return gc


# ══════════════════════════════════════════════════════════════════════════════
# 1.2  N-QUEENS (custom constraints)
#      Bitmask-optimised backtracking, O(n!) pruned
#      Used for: Exam hall seating arrangement
# ══════════════════════════════════════════════════════════════════════════════

class NQueens:
    """
    Classic N-Queens + exam-seating variant with arbitrary constraint functions.
    Bitmask column/diagonal tracking → O(1) per constraint check.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.steps: list[dict] = []
        self.solutions: list[list[int]] = []

    def solve_basic(self, n: int) -> list[list[int]]:
        """Classic N-Queens. Returns list of solutions (board[row] = col)."""
        self.steps = []
        board = [-1] * n

        def bt(row, cols, d1, d2):
            if row == n:
                self.solutions.append(list(board));
                return True
            avail = ((1 << n) - 1) & ~(cols | d1 | d2)
            while avail:
                bit = avail & (-avail);
                avail ^= bit
                col = bit.bit_length() - 1
                board[row] = col
                self.steps.append({'action': 'place', 'row': row, 'col': col, 'board': list(board)})
                if bt(row + 1, cols | bit, (d1 | bit) << 1, (d2 | bit) >> 1):
                    return True
                board[row] = -1
                self.steps.append({'action': 'remove', 'row': row, 'col': col, 'board': list(board)})
            return False

        self.solutions = []
        bt(0, 0, 0, 0)
        return self.solutions

    def count_solutions(self, n: int, limit: int = 100_000) -> int:
        """Count all N-Queens solutions up to limit."""
        count = [0]

        def bt(row, cols, d1, d2):
            if count[0] >= limit: return
            if row == n: count[0] += 1; return
            avail = ((1 << n) - 1) & ~(cols | d1 | d2)
            while avail:
                bit = avail & (-avail);
                avail ^= bit
                bt(row + 1, cols | bit, (d1 | bit) << 1, (d2 | bit) >> 1)

        bt(0, 0, 0, 0)
        return count[0]

    def solve_exam_seating(
            self,
            num_students: int,
            hall_rows: int,
            hall_cols: int,
            dept_map: dict | None = None,
            blocked_seats: set | None = None,
            no_adjacent_same_dept: bool = True,
            faculty_rows: set | None = None,
    ) -> tuple:
        """
        Constrained seating via backtracking.
        Returns (grid[r][c] = student_id | None, steps_list).
        Extra constraints: dept separation, blocked seats, faculty rows.
        """
        blocked_seats = blocked_seats or set()
        faculty_rows = faculty_rows or set()
        dept_map = dept_map or {}

        all_seats = [(r, c) for r in range(hall_rows) if r not in faculty_rows
                     for c in range(hall_cols) if (r, c) not in blocked_seats]

        if len(all_seats) < num_students:
            return None, []

        grid = [[None] * hall_cols for _ in range(hall_rows)]
        self.steps = []

        def neighbors(r, c):
            return [(r + dr, c + dc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    if 0 <= r + dr < hall_rows and 0 <= c + dc < hall_cols]

        def is_valid(r, c, sid):
            if grid[r][c] is not None or (r, c) in blocked_seats: return False
            if no_adjacent_same_dept and dept_map:
                sdept = dept_map.get(sid, 'X')
                for nr, nc in neighbors(r, c):
                    nb = grid[nr][nc]
                    if nb is not None and dept_map.get(nb, 'Y') == sdept:
                        return False
            return True

        def bt(seat_idx, student_idx):
            if student_idx == num_students: return True
            if seat_idx >= len(all_seats):  return False
            r, c = all_seats[seat_idx]
            if is_valid(r, c, student_idx):
                grid[r][c] = student_idx
                self.steps.append({'action': 'place', 'row': r, 'col': c,
                                   'student': student_idx,
                                   'dept': dept_map.get(student_idx, '?'),
                                   'grid_snapshot': [row[:] for row in grid]})
                if bt(seat_idx + 1, student_idx + 1): return True
                grid[r][c] = None
                self.steps.append({'action': 'remove', 'row': r, 'col': c,
                                   'student': student_idx,
                                   'grid_snapshot': [row[:] for row in grid]})
            return bt(seat_idx + 1, student_idx)

        bt(0, 0)
        return grid, self.steps


# ══════════════════════════════════════════════════════════════════════════════
# 1.3  MIN-HEAP  (custom, no heapq)
#      O(log n) push/pop via sift-up / sift-down
#      Used for: Assignment priority queue
# ══════════════════════════════════════════════════════════════════════════════

class MinHeap:
    """
    Binary min-heap implemented on a Python list.
    Tie-breaking via insertion counter to guarantee FIFO order on equal priorities.
    """

    def __init__(self):
        self.heap: list[tuple] = []  # (priority, counter, data)
        self._ctr = 0
        self.steps: list[dict] = []

    # ── indexing helpers ────────────────────────────────────────────────────
    @staticmethod
    def _parent(i):
        return (i - 1) // 2

    @staticmethod
    def _left(i):
        return 2 * i + 1

    @staticmethod
    def _right(i):
        return 2 * i + 2

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    # ── core operations ─────────────────────────────────────────────────────
    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self.heap[i][0] < self.heap[p][0]:
                self._swap(i, p);
                i = p
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            s = i
            l, r = self._left(i), self._right(i)
            if l < n and self.heap[l][0] < self.heap[s][0]: s = l
            if r < n and self.heap[r][0] < self.heap[s][0]: s = r
            if s != i:
                self._swap(i, s); i = s
            else:
                break

    def push(self, priority: float, data):
        self.heap.append((priority, self._ctr, data))
        self._ctr += 1
        self._sift_up(len(self.heap) - 1)
        self.steps.append({'op': 'push', 'priority': priority,
                           'data': str(data)[:40], 'size': len(self.heap)})

    def pop(self) -> tuple:
        if not self.heap: raise IndexError("Heap is empty")
        self._swap(0, len(self.heap) - 1)
        priority, _, data = self.heap.pop()
        if self.heap: self._sift_down(0)
        self.steps.append({'op': 'pop', 'priority': priority,
                           'data': str(data)[:40], 'size': len(self.heap)})
        return priority, data

    def peek(self) -> tuple:
        if not self.heap: raise IndexError("Heap is empty")
        p, _, d = self.heap[0];
        return p, d

    def heapify(self, items: list):
        """Build heap from [(priority, data)] in O(n)."""
        self.heap = [(p, i, d) for i, (p, d) in enumerate(items)]
        self._ctr = len(items)
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    def get_sorted(self) -> list:
        """Return all items sorted by priority without modifying original."""
        tmp = MinHeap();
        tmp.heap = list(self.heap)
        result = []
        while tmp.heap: result.append(tmp.pop())
        return result

    def size(self):
        return len(self.heap)

    def is_empty(self):
        return not self.heap


# ══════════════════════════════════════════════════════════════════════════════
# 1.4  PRIORITY QUEUE  (wraps MinHeap with domain-specific scoring)
#      Priority score = deadline_weight/days + difficulty_weight*diff
#      Used for: Assignment scheduling order
# ══════════════════════════════════════════════════════════════════════════════

class PriorityQueue:
    """Domain-specific priority queue on top of MinHeap."""

    def __init__(self, deadline_weight: float = 0.6, difficulty_weight: float = 0.4):
        self.heap = MinHeap()
        self.dw = deadline_weight
        self.fw = difficulty_weight

    def compute_priority(self, days_left: int, difficulty: int, workload: int = 0) -> float:
        """Lower return value → higher urgency (min-heap picks it first)."""
        urgency = 1.0 / (days_left + 1)
        diff_score = difficulty / 10.0
        load_bonus = workload / 20.0
        score = self.dw * urgency + self.fw * diff_score - 0.1 * load_bonus
        return -score  # negate so min-heap serves highest-urgency first

    def add_assignment(self, a: dict):
        pri = self.compute_priority(a['days_left'], a['difficulty'], a.get('workload', 0))
        self.heap.push(pri, a)

    def get_next(self) -> dict:
        _, data = self.heap.pop();
        return data

    def get_schedule(self) -> list:
        return [d for _, d in self.heap.get_sorted()]

    def size(self): return self.heap.size()


# ══════════════════════════════════════════════════════════════════════════════
# 1.5  TOPOLOGICAL SORT
#      Kahn's (BFS) + DFS variant, cycle detection
#      Used for: Course prerequisites, assignment dependencies
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalSort:
    """
    DAG topological ordering.
    Kahn's: O(V+E).  DFS: O(V+E).
    Cycle detection: if output size < V → cycle exists.
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.adj = defaultdict(list)
        self.in_degree = [0] * num_nodes
        self.labels: dict[int, str] = {}
        self.steps: list[dict] = []

    def add_edge(self, u: int, v: int):
        """u must come before v."""
        self.adj[u].append(v)
        self.in_degree[v] += 1

    def set_label(self, node: int, label: str):
        self.labels[node] = label

    def kahns(self) -> tuple[list, bool]:
        """
        Kahn's BFS algorithm.
        Returns (topological_order, has_cycle).
        """
        self.steps = []
        in_deg = list(self.in_degree)
        queue = deque(i for i in range(self.num_nodes) if in_deg[i] == 0)
        order: list[int] = []

        self.steps.append({'phase': 'init', 'queue': list(queue),
                           'in_degrees': list(in_deg), 'order': []})
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in self.adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0: queue.append(nb)
            self.steps.append({'phase': 'process', 'node': node,
                               'label': self.labels.get(node, f'N{node}'),
                               'queue': list(queue),
                               'in_degrees': list(in_deg),
                               'order': list(order)})

        return order, len(order) != self.num_nodes

    def dfs_topo(self) -> tuple[list, bool]:
        """DFS-based topological sort with cycle detection via grey/black colouring."""
        self.steps = []
        state = [0] * self.num_nodes  # 0=unvisited, 1=in-stack, 2=done
        order: list[int] = []
        has_cycle = [False]

        def dfs(node):
            if has_cycle[0]: return
            state[node] = 1
            for nb in self.adj[node]:
                if state[nb] == 1:   has_cycle[0] = True; return
                if state[nb] == 0:   dfs(nb)
            state[node] = 2
            order.append(node)

        for i in range(self.num_nodes):
            if state[i] == 0: dfs(i)
        order.reverse()
        return order, has_cycle[0]

    def get_levels(self) -> list[list]:
        """Return nodes grouped by BFS depth level (for layered visualisation)."""
        _, has_cycle = self.kahns()
        if has_cycle: return []
        in_deg = list(self.in_degree)
        lv_map: dict[int, int] = {}
        queue = deque()
        for i in range(self.num_nodes):
            if in_deg[i] == 0: lv_map[i] = 0; queue.append(i)
        while queue:
            node = queue.popleft()
            for nb in self.adj[node]:
                in_deg[nb] -= 1
                lv_map[nb] = max(lv_map.get(nb, 0), lv_map[node] + 1)
                if in_deg[nb] == 0: queue.append(nb)
        max_lv = max(lv_map.values(), default=0)
        levels = [[] for _ in range(max_lv + 1)]
        for node, lv in lv_map.items():
            levels[lv].append((node, self.labels.get(node, f'N{node}')))
        return levels

    @staticmethod
    def from_assignments(assignments: list[dict]) -> 'TopologicalSort':
        """Build from [{id, name, prerequisites:[id]}]."""
        n = len(assignments)
        ts = TopologicalSort(n)
        id2idx = {a['id']: i for i, a in enumerate(assignments)}
        for i, a in enumerate(assignments):
            ts.set_label(i, a['name'])
            for pre_id in a.get('prerequisites', []):
                if pre_id in id2idx:
                    ts.add_edge(id2idx[pre_id], i)
        return ts


# ══════════════════════════════════════════════════════════════════════════════
# 1.6  BLOOM FILTER
#      Double-hashing, bit-array via Python int, O(k) operations
#      Used for: Fast student availability / duplicate assignment checks
# ══════════════════════════════════════════════════════════════════════════════

class BloomFilter:
    """
    Probabilistic set-membership structure.
    False negatives: impossible.
    False positives: controlled via capacity + fp_rate.
    Memory: O(m) bits  |  Time: O(k) per op  (k = num hash functions)
    """

    def __init__(self, capacity: int = 1000, false_positive_rate: float = 0.01):
        self.capacity = capacity
        self.fp_rate = false_positive_rate
        # Optimal m = -n*ln(p) / (ln2)²
        self.bit_size = max(1, int(-capacity * math.log(false_positive_rate) / math.log(2) ** 2))
        # Optimal k = (m/n) * ln2
        self.num_hashes = max(1, int((self.bit_size / capacity) * math.log(2)))
        self.bit_array = 0  # Python int used as bit-array (arbitrary width)
        self.count = 0
        self.operations: list[dict] = []

    def _hashes(self, item: str) -> list[int]:
        """Double-hashing: h_i(x) = (h1 + i*h2) % m."""
        raw = str(item).encode()
        h1, h2 = 0, 5381
        for b in raw:
            h1 = (h1 * 31 + b) % self.bit_size
            h2 = ((h2 << 5) + h2 + b) % self.bit_size
        if h2 == 0: h2 = 1
        return [(h1 + i * h2) % self.bit_size for i in range(self.num_hashes)]

    def add(self, item: str):
        positions = self._hashes(item)
        for p in positions: self.bit_array |= (1 << p)
        self.count += 1
        self.operations.append({'op': 'add', 'item': item,
                                'positions': positions, 'count': self.count})

    def contains(self, item: str) -> bool:
        """Probabilistic membership test."""
        positions = self._hashes(item)
        result = all((self.bit_array >> p) & 1 for p in positions)
        self.operations.append({'op': 'check', 'item': item,
                                'positions': positions, 'result': result})
        return result

    def current_fp_rate(self) -> float:
        fill = bin(self.bit_array).count('1') / self.bit_size
        return fill ** self.num_hashes

    def info(self) -> dict:
        set_bits = bin(self.bit_array).count('1')
        return {
            'bit_size': self.bit_size,
            'num_hashes': self.num_hashes,
            'items_added': self.count,
            'bits_set': set_bits,
            'fill_ratio': set_bits / self.bit_size,
            'false_positive_rate': self.current_fp_rate(),
        }

    def reset(self):
        self.bit_array = 0;
        self.count = 0;
        self.operations = []


class AssignmentAvailabilityChecker:
    """
    Thin wrapper: uses BloomFilter for O(k) fast-path
    + exact set to eliminate false positives on confirmation.
    """

    def __init__(self, capacity: int = 5000):
        self.bf = BloomFilter(capacity=capacity, false_positive_rate=0.005)
        self.exact: set = set()

    def mark_busy(self, student_id: str, date: str):
        key = f"{student_id}:{date}"
        self.bf.add(key);
        self.exact.add(key)

    def is_busy(self, student_id: str, date: str) -> bool:
        key = f"{student_id}:{date}"
        return self.bf.contains(key) and key in self.exact

    def bloom_says(self, student_id: str, date: str) -> bool:
        return self.bf.contains(f"{student_id}:{date}")

    def get_bloom_info(self) -> dict:
        return self.bf.info()


# ══════════════════════════════════════════════════════════════════════════════
# 1.7  UNION-FIND  (Disjoint Set Union)
#      Path compression + union-by-rank → O(α(n)) ≈ O(1) amortised
#      Used for: Grouping same-dept students, connected-component queries
# ══════════════════════════════════════════════════════════════════════════════

class UnionFind:
    """
    DSU with path compression and union by rank.
    α(n) is the inverse Ackermann function — practically constant.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size_arr = [1] * n
        self.n = n
        self.num_components = n
        self.operations: list[dict] = []

    def find(self, x: int) -> int:
        """Find root with full path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if merged."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            self.operations.append({'op': 'union', 'x': x, 'y': y, 'merged': False})
            return False
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        self.size_arr[rx] += self.size_arr[ry]
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1
        self.num_components -= 1
        self.operations.append({'op': 'union', 'x': x, 'y': y, 'merged': True, 'root': rx})
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def component_size(self, x: int) -> int:
        return self.size_arr[self.find(x)]

    def get_components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(self.n): groups[self.find(i)].append(i)
        return dict(groups)


# ══════════════════════════════════════════════════════════════════════════════
# 1.8  BACKTRACKING UTILITIES
#      General CSP solver, permutation / subset enumeration
# ══════════════════════════════════════════════════════════════════════════════

class Backtracking:
    """General-purpose backtracking helpers."""

    def __init__(self):
        self.call_count = 0
        self.backtrack_count = 0

    def reset(self):
        self.call_count = self.backtrack_count = 0

    def permutations(self, items: list) -> list[list]:
        result, used, cur = [], [False] * len(items), []

        def bt():
            self.call_count += 1
            if len(cur) == len(items): result.append(list(cur)); return
            for i, item in enumerate(items):
                if not used[i]:
                    used[i] = True;
                    cur.append(item);
                    bt()
                    cur.pop();
                    used[i] = False;
                    self.backtrack_count += 1

        bt();
        return result

    def subsets(self, items: list) -> list[list]:
        return [[items[i] for i in range(len(items)) if mask & (1 << i)]
                for mask in range(1 << len(items))]

    def solve_csp(self, variables: list, domains: dict, constraints: list) -> dict | None:
        """
        General CSP: variables, domains dict, constraints = [fn(partial_assignment)->bool].
        Constraint functions receive a PARTIAL assignment — they must return True
        when a required variable is not yet assigned (forward checking).
        Example: lambda a: 0 not in a or 1 not in a or a[0] != a[1]
        """
        self.reset()
        assignment: dict = {}

        def consistent(var, val):
            tmp = {**assignment, var: val}
            return all(c(tmp) for c in constraints)

        def bt(idx):
            self.call_count += 1
            if idx == len(variables): return True
            var = variables[idx]
            for val in domains[var]:
                if consistent(var, val):
                    assignment[var] = val
                    if bt(idx + 1): return True
                    del assignment[var];
                    self.backtrack_count += 1
            return False

        return assignment if bt(0) else None

    def stats(self) -> dict:
        return {'calls': self.call_count, 'backtracks': self.backtrack_count,
                'efficiency': round(1 - self.backtrack_count / max(1, self.call_count), 3)}