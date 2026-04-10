"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         VIT UNIVERSITY - STUDENT GRADE MANAGEMENT SYSTEM                        ║
║         DSA Project | Python Implementation                                      ║
║                                                                                  ║
║  DATA STRUCTURES USED:                                                           ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║  1. Linked List       → Student records chain                                    ║
║  2. Binary Search Tree (BST) → Fast student lookup by register number            ║
║  3. Stack             → Undo/Redo operations                                     ║
║  4. Queue             → Exam result processing queue                             ║
║  5. Hash Map (Dict)   → Course & subject lookup                                  ║
║  6. Heap (Priority Queue) → Top/Bottom rankers                                   ║
║  7. Graph (Adjacency List) → Course prerequisites                                ║
║  8. Merge Sort        → Sort students by GPA, name, register number              ║
║  9. Binary Search     → Fast grade lookup                                        ║
║  10. Trie             → Autocomplete for student name search                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import heapq
import os
import pickle
from collections import defaultdict, deque
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: CORE DATA MODELS
# ──────────────────────────────────────────────────────────────────────────────

class Grade:
    """Represents a single subject grade entry."""
    GRADE_POINTS = {
        'S': 10, 'A+': 9, 'A': 8, 'B+': 7,
        'B': 6,  'C':  5, 'D': 4, 'F': 0,
        'N/A': 0
    }

    def __init__(self, subject_code: str, subject_name: str,
                 credits: int, grade_letter: str, semester: int):
        self.subject_code  = subject_code
        self.subject_name  = subject_name
        self.credits       = credits
        self.grade_letter  = grade_letter.upper()
        self.semester      = semester
        self.grade_point   = self.GRADE_POINTS.get(self.grade_letter, 0)
        self.grade_score   = self.grade_point * self.credits   # weighted score

    def __repr__(self):
        return (f"Grade({self.subject_code}, {self.grade_letter}, "
                f"Sem-{self.semester}, Credits={self.credits})")


class Student:
    """Complete student profile with grade history."""

    def __init__(self, reg_no: str, name: str, department: str,
                 batch: int, email: str = ""):
        self.reg_no      = reg_no
        self.name        = name
        self.department  = department
        self.batch       = batch
        self.email       = email
        self.grades: list[Grade] = []          # list of Grade objects
        self.cgpa        = 0.0
        self.total_credits = 0
        self.created_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── GPA Calculation ────────────────────────────────────────────────────

    def calculate_cgpa(self) -> float:
        """Calculates Cumulative GPA using weighted average formula."""
        total_weighted = sum(g.grade_score for g in self.grades)
        total_credits  = sum(g.credits for g in self.grades
                             if g.grade_letter != 'F')
        self.total_credits = total_credits
        self.cgpa = round(total_weighted / total_credits, 2) if total_credits else 0.0
        return self.cgpa

    def get_sgpa(self, semester: int) -> float:
        """Calculates Semester GPA for a specific semester."""
        sem_grades = [g for g in self.grades if g.semester == semester]
        total_weighted = sum(g.grade_score for g in sem_grades)
        total_credits  = sum(g.credits for g in sem_grades
                             if g.grade_letter != 'F')
        return round(total_weighted / total_credits, 2) if total_credits else 0.0

    def add_grade(self, grade: 'Grade'):
        self.grades.append(grade)
        self.calculate_cgpa()

    def get_arrears(self) -> list:
        return [g for g in self.grades if g.grade_letter == 'F']

    def get_subjects_by_semester(self, semester: int) -> list:
        return [g for g in self.grades if g.semester == semester]

    def __repr__(self):
        return f"Student({self.reg_no}, {self.name}, CGPA={self.cgpa})"

    def __lt__(self, other):
        return self.cgpa < other.cgpa


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA STRUCTURE #1 — DOUBLY LINKED LIST (Student Records Chain)
# ──────────────────────────────────────────────────────────────────────────────

class LinkedNode:
    """Node for Doubly Linked List."""
    def __init__(self, student: Student):
        self.student = student
        self.prev    = None
        self.next    = None


class StudentLinkedList:
    """
    Doubly Linked List to maintain ordered student records.
    Supports O(1) insertion at head/tail, O(n) search.
    """

    def __init__(self):
        self.head  = None
        self.tail  = None
        self.size  = 0

    def append(self, student: Student):
        """Insert at tail — O(1)."""
        node = LinkedNode(student)
        if not self.tail:
            self.head = self.tail = node
        else:
            node.prev      = self.tail
            self.tail.next = node
            self.tail      = node
        self.size += 1

    def prepend(self, student: Student):
        """Insert at head — O(1)."""
        node = LinkedNode(student)
        if not self.head:
            self.head = self.tail = node
        else:
            node.next       = self.head
            self.head.prev  = node
            self.head       = node
        self.size += 1

    def delete(self, reg_no: str) -> bool:
        """Remove a student by reg_no — O(n)."""
        current = self.head
        while current:
            if current.student.reg_no == reg_no:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                self.size -= 1
                return True
            current = current.next
        return False

    def find(self, reg_no: str) -> Student | None:
        """Linear search by reg_no — O(n)."""
        current = self.head
        while current:
            if current.student.reg_no == reg_no:
                return current.student
            current = current.next
        return None

    def to_list(self) -> list[Student]:
        result, current = [], self.head
        while current:
            result.append(current.student)
            current = current.next
        return result

    def __len__(self):
        return self.size

    def __iter__(self):
        current = self.head
        while current:
            yield current.student
            current = current.next


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA STRUCTURE #2 — BINARY SEARCH TREE (Fast Lookup)
# ──────────────────────────────────────────────────────────────────────────────

class BSTNode:
    """BST node keyed on reg_no (string comparison)."""
    def __init__(self, student: Student):
        self.student = student
        self.left    = None
        self.right   = None


class StudentBST:
    """
    Binary Search Tree for O(log n) average-case lookup by register number.
    Provides in-order traversal (sorted by reg_no) and range queries.
    """

    def __init__(self):
        self.root = None

    # ── Insert ─────────────────────────────────────────────────────────────
    def insert(self, student: Student):
        self.root = self._insert(self.root, student)

    def _insert(self, node: BSTNode | None, student: Student) -> BSTNode:
        if node is None:
            return BSTNode(student)
        if student.reg_no < node.student.reg_no:
            node.left  = self._insert(node.left, student)
        elif student.reg_no > node.student.reg_no:
            node.right = self._insert(node.right, student)
        else:
            node.student = student          # update existing
        return node

    # ── Search ─────────────────────────────────────────────────────────────
    def search(self, reg_no: str) -> Student | None:
        return self._search(self.root, reg_no)

    def _search(self, node: BSTNode | None, reg_no: str) -> Student | None:
        if node is None:
            return None
        if reg_no == node.student.reg_no:
            return node.student
        if reg_no < node.student.reg_no:
            return self._search(node.left, reg_no)
        return self._search(node.right, reg_no)

    # ── Delete ─────────────────────────────────────────────────────────────
    def delete(self, reg_no: str):
        self.root = self._delete(self.root, reg_no)

    def _delete(self, node: BSTNode | None, reg_no: str) -> BSTNode | None:
        if node is None:
            return None
        if reg_no < node.student.reg_no:
            node.left  = self._delete(node.left, reg_no)
        elif reg_no > node.student.reg_no:
            node.right = self._delete(node.right, reg_no)
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            # In-order successor
            successor = self._min_node(node.right)
            node.student = successor.student
            node.right   = self._delete(node.right, successor.student.reg_no)
        return node

    def _min_node(self, node: BSTNode) -> BSTNode:
        while node.left:
            node = node.left
        return node

    # ── Traversals ─────────────────────────────────────────────────────────
    def inorder(self) -> list[Student]:
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: BSTNode | None, result: list):
        if node:
            self._inorder(node.left, result)
            result.append(node.student)
            self._inorder(node.right, result)

    def range_query(self, low: str, high: str) -> list[Student]:
        """Return all students with reg_no in [low, high]."""
        result = []
        self._range(self.root, low, high, result)
        return result

    def _range(self, node, low, high, result):
        if not node:
            return
        if low < node.student.reg_no:
            self._range(node.left, low, high, result)
        if low <= node.student.reg_no <= high:
            result.append(node.student)
        if high > node.student.reg_no:
            self._range(node.right, low, high, result)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: DATA STRUCTURE #3 — STACK (Undo/Redo Operations)
# ──────────────────────────────────────────────────────────────────────────────

class OperationStack:
    """
    Stack-based Undo/Redo system for grade modifications.
    Each entry stores (operation_type, student_copy) for rollback.
    """

    def __init__(self, max_size: int = 50):
        self._stack   = []
        self.max_size = max_size

    def push(self, operation: str, data):
        if len(self._stack) >= self.max_size:
            self._stack.pop(0)          # Remove oldest
        self._stack.append((operation, data, datetime.now()))

    def pop(self):
        return self._stack.pop() if self._stack else None

    def peek(self):
        return self._stack[-1] if self._stack else None

    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def size(self) -> int:
        return len(self._stack)

    def clear(self):
        self._stack.clear()

    def history(self) -> list:
        return list(reversed(self._stack))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: DATA STRUCTURE #4 — QUEUE (Exam Result Processing)
# ──────────────────────────────────────────────────────────────────────────────

class ResultProcessingQueue:
    """
    FIFO Queue for batch-processing pending exam results.
    Uses collections.deque for O(1) enqueue and dequeue.
    """

    def __init__(self):
        self._queue = deque()

    def enqueue(self, task: dict):
        """Add a result-processing task."""
        task['enqueued_at'] = datetime.now().strftime("%H:%M:%S")
        self._queue.append(task)

    def dequeue(self) -> dict | None:
        return self._queue.popleft() if self._queue else None

    def peek(self) -> dict | None:
        return self._queue[0] if self._queue else None

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def size(self) -> int:
        return len(self._queue)

    def __iter__(self):
        return iter(self._queue)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: DATA STRUCTURE #5 — TRIE (Student Name Autocomplete)
# ──────────────────────────────────────────────────────────────────────────────

class TrieNode:
    def __init__(self):
        self.children  = {}
        self.is_end    = False
        self.reg_nos   = []     # All reg_nos matching this prefix


class Trie:
    """
    Trie for O(m) prefix-based student name autocomplete.
    m = length of search prefix.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, name: str, reg_no: str):
        """Insert student name (lowercase) into Trie."""
        node = self.root
        for ch in name.lower():
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            if reg_no not in node.reg_nos:
                node.reg_nos.append(reg_no)
        node.is_end = True

    def search_prefix(self, prefix: str) -> list[str]:
        """Return all reg_nos whose names start with prefix."""
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return []
            node = node.children[ch]
        return node.reg_nos

    def delete(self, name: str, reg_no: str):
        """Remove a reg_no entry from the trie."""
        node = self.root
        for ch in name.lower():
            if ch not in node.children:
                return
            node = node.children[ch]
            if reg_no in node.reg_nos:
                node.reg_nos.remove(reg_no)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: DATA STRUCTURE #6 — MIN/MAX HEAP (Rankings)
# ──────────────────────────────────────────────────────────────────────────────

class RankingHeap:
    """
    Uses Python's heapq (min-heap) for top/bottom ranker queries.
    For max-heap, values are negated.
    """

    @staticmethod
    def top_n_students(students: list[Student], n: int) -> list[Student]:
        """Return top-n students by CGPA using Max-Heap."""
        # Use negative CGPA for max-heap behaviour
        heap = [(-s.cgpa, s.reg_no, s) for s in students]
        heapq.heapify(heap)
        result = []
        for _ in range(min(n, len(heap))):
            _, _, student = heapq.heappop(heap)
            result.append(student)
        return result

    @staticmethod
    def bottom_n_students(students: list[Student], n: int) -> list[Student]:
        """Return bottom-n students by CGPA using Min-Heap."""
        heap = [(s.cgpa, s.reg_no, s) for s in students]
        heapq.heapify(heap)
        result = []
        for _ in range(min(n, len(heap))):
            _, _, student = heapq.heappop(heap)
            result.append(student)
        return result

    @staticmethod
    def get_rank(students: list[Student], reg_no: str) -> int:
        """Return 1-based rank of a student among all students."""
        sorted_students = sorted(students, key=lambda s: -s.cgpa)
        for idx, s in enumerate(sorted_students):
            if s.reg_no == reg_no:
                return idx + 1
        return -1


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: DATA STRUCTURE #7 — GRAPH (Course Prerequisites)
# ──────────────────────────────────────────────────────────────────────────────

class CourseGraph:
    """
    Directed Graph (Adjacency List) to model course prerequisites.
    Edge A → B means: A is a prerequisite for B.
    Uses DFS-based cycle detection and topological sort.
    """

    def __init__(self):
        self.adj   = defaultdict(list)   # course → [dependent courses]
        self.prereq = defaultdict(list)  # course → [prerequisite courses]
        self.courses = {}                # code → name

    def add_course(self, code: str, name: str):
        self.courses[code] = name
        if code not in self.adj:
            self.adj[code] = []

    def add_prerequisite(self, prereq_code: str, course_code: str):
        """prereq_code must be completed before course_code."""
        self.adj[prereq_code].append(course_code)
        self.prereq[course_code].append(prereq_code)

    def get_prerequisites(self, course_code: str) -> list[str]:
        return self.prereq.get(course_code, [])

    def get_dependents(self, course_code: str) -> list[str]:
        return self.adj.get(course_code, [])

    def can_enroll(self, student: Student, course_code: str) -> tuple[bool, list]:
        """Check if student has completed all prerequisites."""
        completed = {g.subject_code for g in student.grades
                     if g.grade_letter not in ('F', 'N/A')}
        missing = [p for p in self.prereq[course_code] if p not in completed]
        return (len(missing) == 0), missing

    def topological_sort(self) -> list[str]:
        """Kahn's Algorithm — BFS-based topological sort."""
        in_degree = defaultdict(int)
        for node in self.adj:
            for neighbor in self.adj[node]:
                in_degree[neighbor] += 1

        queue  = deque([c for c in self.courses if in_degree[c] == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return result

    def has_cycle(self) -> bool:
        """DFS-based cycle detection."""
        visited, rec_stack = set(), set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        return any(dfs(c) for c in self.courses if c not in visited)

    def bfs_reachable(self, start_code: str) -> list[str]:
        """BFS from a course — all courses unlocked after completing it."""
        visited = set()
        queue   = deque([start_code])
        result  = []
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            result.append(node)
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        return result[1:]    # exclude start itself


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: DATA STRUCTURE #8 — SORTING ALGORITHMS
# ──────────────────────────────────────────────────────────────────────────────

class Sorter:
    """Collection of sorting algorithms for student lists."""

    # ── Merge Sort ─────────────────────────────────────────────────────────
    @staticmethod
    def merge_sort(arr: list[Student], key=lambda s: s.cgpa,
                   reverse: bool = False) -> list[Student]:
        """O(n log n) — stable sort by any key."""
        if len(arr) <= 1:
            return arr
        mid   = len(arr) // 2
        left  = Sorter.merge_sort(arr[:mid],  key, reverse)
        right = Sorter.merge_sort(arr[mid:], key, reverse)
        return Sorter._merge(left, right, key, reverse)

    @staticmethod
    def _merge(left, right, key, reverse):
        result, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            lv, rv = key(left[i]), key(right[j])
            cond = lv >= rv if reverse else lv <= rv
            if cond:
                result.append(left[i]);  i += 1
            else:
                result.append(right[j]); j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    # ── Quick Sort ─────────────────────────────────────────────────────────
    @staticmethod
    def quick_sort(arr: list, key=lambda s: s.cgpa,
                   reverse: bool = False) -> list:
        """O(n log n) average — sorts in-place variant."""
        if len(arr) <= 1:
            return arr
        pivot = key(arr[len(arr) // 2])
        left  = [x for x in arr if key(x) < pivot]
        mid   = [x for x in arr if key(x) == pivot]
        right = [x for x in arr if key(x) > pivot]
        if reverse:
            return (Sorter.quick_sort(right, key, reverse) + mid +
                    Sorter.quick_sort(left,  key, reverse))
        return (Sorter.quick_sort(left,  key, reverse) + mid +
                Sorter.quick_sort(right, key, reverse))

    # ── Insertion Sort (good for small n) ──────────────────────────────────
    @staticmethod
    def insertion_sort(arr: list[Student], key=lambda s: s.name) -> list[Student]:
        """O(n²) — stable, efficient for small or nearly-sorted data."""
        arr = arr[:]
        for i in range(1, len(arr)):
            current = arr[i]
            j = i - 1
            while j >= 0 and key(arr[j]) > key(current):
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = current
        return arr


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: DATA STRUCTURE #9 — BINARY SEARCH
# ──────────────────────────────────────────────────────────────────────────────

class BinarySearch:
    """Binary Search utilities for sorted student arrays."""

    @staticmethod
    def search_by_cgpa(sorted_students: list[Student],
                       target_cgpa: float) -> int:
        """
        Returns index of a student with target_cgpa in sorted (asc) list.
        Returns -1 if not found. O(log n).
        """
        low, high = 0, len(sorted_students) - 1
        while low <= high:
            mid = (low + high) // 2
            if sorted_students[mid].cgpa == target_cgpa:
                return mid
            elif sorted_students[mid].cgpa < target_cgpa:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    @staticmethod
    def search_by_name(sorted_students: list[Student],
                       target_name: str) -> int:
        """Binary search on name-sorted student list. O(log n)."""
        low, high = 0, len(sorted_students) - 1
        target = target_name.lower()
        while low <= high:
            mid  = (low + high) // 2
            curr = sorted_students[mid].name.lower()
            if curr == target:
                return mid
            elif curr < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    @staticmethod
    def lower_bound(sorted_students: list[Student],
                    min_cgpa: float) -> int:
        """Return first index where cgpa >= min_cgpa."""
        low, high = 0, len(sorted_students)
        while low < high:
            mid = (low + high) // 2
            if sorted_students[mid].cgpa < min_cgpa:
                low = mid + 1
            else:
                high = mid
        return low

    @staticmethod
    def upper_bound(sorted_students: list[Student],
                    max_cgpa: float) -> int:
        """Return last index where cgpa <= max_cgpa."""
        low, high = 0, len(sorted_students)
        while low < high:
            mid = (low + high) // 2
            if sorted_students[mid].cgpa <= max_cgpa:
                low = mid + 1
            else:
                high = mid
        return low - 1


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: HASH MAP — Course Master & Department Registry
# ──────────────────────────────────────────────────────────────────────────────

class CourseRegistry:
    """
    Hash Map (Python dict) for O(1) average course lookups.
    Stores subject metadata indexed by subject_code.
    """

    def __init__(self):
        self._courses: dict[str, dict] = {}

    def add_course(self, code: str, name: str, credits: int,
                   department: str, semester: int):
        self._courses[code] = {
            'code':       code,
            'name':       name,
            'credits':    credits,
            'department': department,
            'semester':   semester,
        }

    def get(self, code: str) -> dict | None:
        return self._courses.get(code)

    def exists(self, code: str) -> bool:
        return code in self._courses

    def all_codes(self) -> list[str]:
        return list(self._courses.keys())

    def by_department(self, dept: str) -> list[dict]:
        return [c for c in self._courses.values()
                if c['department'] == dept]

    def by_semester(self, sem: int) -> list[dict]:
        return [c for c in self._courses.values()
                if c['semester'] == sem]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12: MAIN MANAGEMENT SYSTEM (Orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

class VITGradeManagementSystem:
    """
    Central orchestrator combining all data structures.
    Provides a complete CLI-based management interface.
    """

    DEPARTMENTS = ['CSE', 'ECE', 'EEE', 'MECH', 'CIVIL', 'IT', 'AIDS', 'AIML']
    SAVE_FILE   = 'vit_gms_data.pkl'

    def __init__(self):
        # Core data structures
        self.linked_list  = StudentLinkedList()
        self.bst          = StudentBST()
        self.trie         = Trie()
        self.undo_stack   = OperationStack()
        self.redo_stack   = OperationStack()
        self.result_queue = ResultProcessingQueue()
        self.course_graph = CourseGraph()
        self.course_registry = CourseRegistry()
        self.ranking_heap = RankingHeap()

        # Fast lookup dictionaries
        self._students: dict[str, Student] = {}   # reg_no → Student

        # Statistics cache
        self._dept_stats: dict[str, list] = defaultdict(list)

        # Load demo data
        self._setup_demo_courses()
        self._load_demo_students()

    # ── Setup ──────────────────────────────────────────────────────────────
    def _setup_demo_courses(self):
        """Pre-populate VIT-style course registry and prerequisite graph."""
        courses = [
            ("MAT1001", "Engineering Mathematics I",     4, "ALL",  1),
            ("PHY1001", "Engineering Physics",            3, "ALL",  1),
            ("CHY1001", "Engineering Chemistry",          3, "ALL",  1),
            ("CSE1001", "Problem Solving and Python",     4, "CSE",  1),
            ("CSE1002", "Problem Solving and C",          4, "CSE",  1),
            ("MAT1002", "Engineering Mathematics II",     4, "ALL",  2),
            ("CSE1003", "Data Structures and Algorithms", 4, "CSE",  2),
            ("CSE1004", "Object Oriented Programming",    4, "CSE",  2),
            ("CSE2001", "Database Management Systems",    4, "CSE",  3),
            ("CSE2002", "Operating Systems",              4, "CSE",  3),
            ("CSE2003", "Computer Networks",              4, "CSE",  4),
            ("CSE2004", "Software Engineering",           3, "CSE",  4),
            ("CSE3001", "Machine Learning",               4, "CSE",  5),
            ("CSE3002", "Deep Learning",                  4, "CSE",  6),
            ("CSE4001", "Capstone Project",               6, "CSE",  8),
        ]
        for code, name, cred, dept, sem in courses:
            self.course_registry.add_course(code, name, cred, dept, sem)
            self.course_graph.add_course(code, name)

        # Prerequisites
        prereqs = [
            ("CSE1001", "CSE1003"),
            ("CSE1002", "CSE1003"),
            ("MAT1001", "MAT1002"),
            ("CSE1003", "CSE2001"),
            ("CSE1003", "CSE2002"),
            ("CSE2001", "CSE3001"),
            ("CSE2002", "CSE2003"),
            ("CSE3001", "CSE3002"),
            ("CSE3002", "CSE4001"),
        ]
        for pre, course in prereqs:
            self.course_graph.add_prerequisite(pre, course)

    def _load_demo_students(self):
        """Create 10 sample VIT students with grades."""
        demo_data = [
            ("22BCE0001", "Arjun Krishnamurthy",  "CSE",  2022,
             [("MAT1001",4,'S',1),("PHY1001",3,'A+',1),("CSE1001",4,'S',1),
              ("MAT1002",4,'A',2),("CSE1003",4,'S',2),("CSE1004",4,'A+',2),
              ("CSE2001",4,'A',3),("CSE2002",4,'A+',3)]),

            ("22BCE0002", "Priya Venkataraman",   "CSE",  2022,
             [("MAT1001",4,'A+',1),("PHY1001",3,'A',1),("CSE1001",4,'A',1),
              ("MAT1002",4,'B+',2),("CSE1003",4,'A+',2),("CSE1004",4,'A',2),
              ("CSE2001",4,'B+',3),("CSE2002",4,'A',3)]),

            ("22BCE0003", "Rohit Sharma",          "CSE",  2022,
             [("MAT1001",4,'B+',1),("PHY1001",3,'B',1),("CSE1001",4,'A',1),
              ("MAT1002",4,'B',2),("CSE1003",4,'B+',2),("CSE1004",4,'B',2),
              ("CSE2001",4,'C',3),("CSE2002",4,'B+',3)]),

            ("22BCE0004", "Ananya Iyer",           "CSE",  2022,
             [("MAT1001",4,'S',1),("PHY1001",3,'S',1),("CSE1001",4,'A+',1),
              ("MAT1002",4,'S',2),("CSE1003",4,'A+',2),("CSE1004",4,'S',2),
              ("CSE2001",4,'S',3),("CSE2002",4,'A+',3)]),

            ("22BCE0005", "Karthik Sundaram",      "CSE",  2022,
             [("MAT1001",4,'A',1),("PHY1001",3,'B+',1),("CSE1001",4,'B+',1),
              ("MAT1002",4,'A+',2),("CSE1003",4,'A',2),("CSE1004",4,'A+',2),
              ("CSE2001",4,'A+',3),("CSE2002",4,'A',3)]),

            ("22ECE0001", "Divya Rajendran",       "ECE",  2022,
             [("MAT1001",4,'A',1),("PHY1001",3,'A+',1),("CHY1001",3,'A',1),
              ("MAT1002",4,'A+',2),("CSE1003",4,'B+',2)]),

            ("22ECE0002", "Suresh Balakrishnan",   "ECE",  2022,
             [("MAT1001",4,'B',1),("PHY1001",3,'B+',1),("CHY1001",3,'B',1),
              ("MAT1002",4,'B+',2),("CSE1003",4,'C',2)]),

            ("22AIDS001", "Meera Nair",             "AIDS", 2022,
             [("MAT1001",4,'S',1),("PHY1001",3,'A+',1),("CSE1001",4,'S',1),
              ("MAT1002",4,'A+',2),("CSE1003",4,'S',2),("CSE3001",4,'A+',3)]),

            ("21BCE0010", "Varun Pillai",           "CSE",  2021,
             [("MAT1001",4,'A',1),("PHY1001",3,'A',1),("CSE1001",4,'A+',1),
              ("MAT1002",4,'A+',2),("CSE1003",4,'A',2),("CSE1004",4,'A+',2),
              ("CSE2001",4,'A',3),("CSE2002",4,'A+',3),
              ("CSE2003",4,'A',4),("CSE2004",3,'A+',4)]),

            ("21BCE0011", "Lakshmi Subramanian",   "CSE",  2021,
             [("MAT1001",4,'F',1),("PHY1001",3,'B',1),("CSE1001",4,'A',1),
              ("MAT1001",4,'C',1),   # retake after fail
              ("MAT1002",4,'B+',2),("CSE1003",4,'B',2),("CSE1004",4,'B+',2),
              ("CSE2001",4,'B',3),("CSE2002",4,'B+',3)]),
        ]

        for reg, name, dept, batch, grades in demo_data:
            student = Student(reg, name, dept, batch,
                              f"{reg.lower()}@vitstudent.ac.in")
            for code, cred, letter, sem in grades:
                course = self.course_registry.get(code)
                cname  = course['name'] if course else code
                student.add_grade(Grade(code, cname, cred, letter, sem))
            self._register_student(student)

    def _register_student(self, student: Student):
        """Internal: add student to all data structures."""
        self._students[student.reg_no] = student
        self.linked_list.append(student)
        self.bst.insert(student)
        self.trie.insert(student.name, student.reg_no)
        self._dept_stats[student.department].append(student)

    # ── Public CRUD Operations ─────────────────────────────────────────────

    def add_student(self, reg_no: str, name: str, department: str,
                    batch: int, email: str = "") -> tuple[bool, str]:
        """Add a new student to the system."""
        if reg_no in self._students:
            return False, f"Student {reg_no} already exists."
        if department.upper() not in self.DEPARTMENTS:
            return False, f"Invalid department. Choose from {self.DEPARTMENTS}"
        student = Student(reg_no, name, department.upper(), batch, email)
        self._register_student(student)
        # Push to undo stack
        self.undo_stack.push("ADD_STUDENT", {"reg_no": reg_no})
        return True, f"✅ Student {name} ({reg_no}) added successfully."

    def remove_student(self, reg_no: str) -> tuple[bool, str]:
        """Remove a student (with undo support)."""
        student = self._students.get(reg_no)
        if not student:
            return False, f"Student {reg_no} not found."
        # Save snapshot for undo
        self.undo_stack.push("REMOVE_STUDENT", student)
        del self._students[reg_no]
        self.linked_list.delete(reg_no)
        self.bst.delete(reg_no)
        self.trie.delete(student.name, reg_no)
        self._dept_stats[student.department] = [
            s for s in self._dept_stats[student.department]
            if s.reg_no != reg_no
        ]
        return True, f"🗑️  Student {student.name} ({reg_no}) removed."

    def add_grade(self, reg_no: str, subject_code: str,
                  grade_letter: str, semester: int) -> tuple[bool, str]:
        """Add/update a grade for a student."""
        student = self._students.get(reg_no)
        if not student:
            return False, f"Student {reg_no} not found."
        course = self.course_registry.get(subject_code)
        if not course:
            return False, f"Subject {subject_code} not in registry."
        # Save old CGPA for undo
        self.undo_stack.push("ADD_GRADE", {
            "reg_no":   reg_no,
            "old_cgpa": student.cgpa,
            "old_grades": list(student.grades)
        })
        grade = Grade(subject_code, course['name'],
                      course['credits'], grade_letter, semester)
        student.add_grade(grade)
        self.bst.insert(student)   # Update BST with new CGPA
        return True, (f"✅ Grade {grade_letter} added for {subject_code} "
                      f"to {student.name}. New CGPA: {student.cgpa}")

    def get_student(self, reg_no: str) -> Student | None:
        """O(log n) lookup using BST."""
        return self.bst.search(reg_no)

    def search_by_name_prefix(self, prefix: str) -> list[Student]:
        """Trie-powered prefix search — O(m) where m = prefix length."""
        reg_nos = self.trie.search_prefix(prefix)
        return [self._students[r] for r in reg_nos if r in self._students]

    def undo_last_operation(self) -> str:
        """Undo the last operation using the undo stack."""
        if self.undo_stack.is_empty():
            return "Nothing to undo."
        op, data, ts = self.undo_stack.pop()
        if op == "ADD_STUDENT":
            reg_no = data["reg_no"]
            if reg_no in self._students:
                student = self._students[reg_no]
                del self._students[reg_no]
                self.linked_list.delete(reg_no)
                self.bst.delete(reg_no)
            return f"↩️  Undone: ADD_STUDENT {reg_no}"
        elif op == "ADD_GRADE":
            reg_no = data["reg_no"]
            if reg_no in self._students:
                student = self._students[reg_no]
                student.grades = data["old_grades"]
                student.calculate_cgpa()
                self.bst.insert(student)
            return f"↩️  Undone: ADD_GRADE for {reg_no}. CGPA restored."
        elif op == "REMOVE_STUDENT":
            student = data
            self._register_student(student)
            return f"↩️  Undone: REMOVE_STUDENT {student.reg_no}"
        return f"↩️  Undone operation: {op}"

    # ── Analytics & Reports ────────────────────────────────────────────────

    def get_top_students(self, n: int = 5, department: str = None) -> list[Student]:
        """Top-n students by CGPA using Max-Heap."""
        pool = (self._dept_stats.get(department.upper(), [])
                if department else list(self._students.values()))
        return self.ranking_heap.top_n_students(pool, n)

    def get_bottom_students(self, n: int = 5) -> list[Student]:
        return self.ranking_heap.bottom_n_students(
            list(self._students.values()), n)

    def get_cgpa_range(self, low: float, high: float) -> list[Student]:
        """Binary search-based range query."""
        sorted_s = Sorter.merge_sort(list(self._students.values()),
                                     key=lambda s: s.cgpa)
        lo = BinarySearch.lower_bound(sorted_s, low)
        hi = BinarySearch.upper_bound(sorted_s, high)
        return sorted_s[lo: hi + 1]

    def department_report(self, dept: str) -> dict:
        """Statistical report for a department."""
        students = self._dept_stats.get(dept.upper(), [])
        if not students:
            return {}
        cgpas = [s.cgpa for s in students]
        return {
            "department":    dept.upper(),
            "total_students": len(students),
            "avg_cgpa":      round(sum(cgpas) / len(cgpas), 2),
            "max_cgpa":      max(cgpas),
            "min_cgpa":      min(cgpas),
            "distinction":   sum(1 for c in cgpas if c >= 9.0),
            "first_class":   sum(1 for c in cgpas if 8.0 <= c < 9.0),
            "second_class":  sum(1 for c in cgpas if 6.5 <= c < 8.0),
            "pass":          sum(1 for c in cgpas if 5.0 <= c < 6.5),
            "fail":          sum(1 for c in cgpas if c < 5.0),
        }

    def all_students_sorted(self, key: str = "cgpa",
                            reverse: bool = True) -> list[Student]:
        """Merge sort by cgpa, name, or reg_no."""
        keys = {
            "cgpa":   lambda s: s.cgpa,
            "name":   lambda s: s.name.lower(),
            "reg_no": lambda s: s.reg_no,
            "batch":  lambda s: s.batch,
        }
        return Sorter.merge_sort(list(self._students.values()),
                                 key=keys.get(key, keys["cgpa"]),
                                 reverse=reverse)

    def check_course_eligibility(self, reg_no: str,
                                  course_code: str) -> tuple[bool, str]:
        """Graph-based prerequisite check."""
        student = self._students.get(reg_no)
        if not student:
            return False, "Student not found."
        ok, missing = self.course_graph.can_enroll(student, course_code)
        if ok:
            return True, f"✅ {student.name} is eligible for {course_code}."
        return False, (f"❌ Missing prerequisites: {', '.join(missing)}")

    def enqueue_result_processing(self, reg_no: str, subject_code: str,
                                   grade: str, semester: int):
        """Add a result to the processing queue."""
        self.result_queue.enqueue({
            "reg_no":       reg_no,
            "subject_code": subject_code,
            "grade":        grade,
            "semester":     semester,
        })

    def process_result_queue(self) -> list[str]:
        """Drain the result queue and apply grades."""
        messages = []
        while not self.result_queue.is_empty():
            task = self.result_queue.dequeue()
            ok, msg = self.add_grade(
                task['reg_no'], task['subject_code'],
                task['grade'],  task['semester']
            )
            messages.append(msg)
        return messages

    # ── Display Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _divider(char: str = "─", width: int = 70):
        print(char * width)

    @staticmethod
    def _header(title: str, width: int = 70):
        VITGradeManagementSystem._divider("═", width)
        print(f"  {title.upper()}")
        VITGradeManagementSystem._divider("═", width)

    def print_student_card(self, student: Student):
        """Pretty-print a student's complete profile."""
        self._header(f"Student Profile — {student.reg_no}")
        print(f"  Name          : {student.name}")
        print(f"  Department    : {student.department}")
        print(f"  Batch         : {student.batch}")
        print(f"  Email         : {student.email}")
        print(f"  CGPA          : {student.cgpa} / 10.0")
        print(f"  Total Credits : {student.total_credits}")
        arrears = student.get_arrears()
        print(f"  Arrears       : {len(arrears)}"
              + (f"  ({', '.join(g.subject_code for g in arrears)})"
                 if arrears else ""))
        self._divider()
        print(f"  {'Sem':<5}{'Code':<12}{'Subject':<35}{'Cr':<5}{'Grade':<6}{'GP'}")
        self._divider()
        for g in student.grades:
            print(f"  {g.semester:<5}{g.subject_code:<12}"
                  f"{g.subject_name[:33]:<35}{g.credits:<5}"
                  f"{g.grade_letter:<6}{g.grade_point}")
        self._divider()
        semesters = sorted(set(g.semester for g in student.grades))
        print("  Semester GPA Summary:")
        for sem in semesters:
            sgpa = student.get_sgpa(sem)
            bar  = "█" * int(sgpa) + "░" * (10 - int(sgpa))
            print(f"    Sem {sem}: {bar} {sgpa}")
        self._divider("═")

    def print_rankings(self, n: int = 5):
        self._header(f"Top {n} Students — VIT University")
        top = self.get_top_students(n)
        for rank, s in enumerate(top, 1):
            star = "⭐" if rank == 1 else "  "
            print(f"  {star} #{rank:<4} {s.reg_no:<14} {s.name:<28} "
                  f"CGPA: {s.cgpa:<6} [{s.department}]")
        self._divider("═")

    def print_department_report(self, dept: str):
        r = self.department_report(dept)
        if not r:
            print(f"No data for department: {dept}")
            return
        self._header(f"Department Report — {dept}")
        print(f"  Total Students : {r['total_students']}")
        print(f"  Average CGPA   : {r['avg_cgpa']}")
        print(f"  Highest CGPA   : {r['max_cgpa']}")
        print(f"  Lowest CGPA    : {r['min_cgpa']}")
        self._divider()
        print(f"  Grade Distribution:")
        print(f"    Distinction (≥9.0) : {'█' * r['distinction']} {r['distinction']}")
        print(f"    First Class (8–9)  : {'█' * r['first_class']} {r['first_class']}")
        print(f"    Second Class (6.5) : {'█' * r['second_class']} {r['second_class']}")
        print(f"    Pass (5–6.5)       : {'█' * r['pass']} {r['pass']}")
        print(f"    Fail (<5.0)        : {'█' * r['fail']} {r['fail']}")
        self._divider("═")

    def print_course_graph_info(self, course_code: str):
        self._header(f"Course Info — {course_code}")
        course = self.course_registry.get(course_code)
        if course:
            print(f"  Name      : {course['name']}")
            print(f"  Credits   : {course['credits']}")
            print(f"  Dept      : {course['department']}")
            print(f"  Semester  : {course['semester']}")
        prereqs = self.course_graph.get_prerequisites(course_code)
        unlocks = self.course_graph.get_dependents(course_code)
        print(f"  Prerequisites : "
              + (", ".join(prereqs) if prereqs else "None"))
        print(f"  Unlocks       : "
              + (", ".join(unlocks) if unlocks else "None"))
        order = self.course_graph.topological_sort()
        if course_code in order:
            print(f"  Study Order   : #{order.index(course_code) + 1} in curriculum")
        self._divider("═")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13: CLI MENU INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

class CLI:
    """Command-Line Interface for VIT Grade Management System."""

    def __init__(self):
        self.gms = VITGradeManagementSystem()

    def _clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _pause(self):
        input("\n  Press Enter to continue...")

    def banner(self):
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║      ██╗   ██╗██╗████████╗    ██████╗ ███╗   ███╗███████╗           ║
║      ██║   ██║██║╚══██╔══╝   ██╔════╝ ████╗ ████║██╔════╝           ║
║      ██║   ██║██║   ██║      ██║  ███╗██╔████╔██║███████╗           ║
║      ╚██╗ ██╔╝██║   ██║      ██║   ██║██║╚██╔╝██║╚════██║           ║
║       ╚████╔╝ ██║   ██║      ╚██████╔╝██║ ╚═╝ ██║███████║           ║
║        ╚═══╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝     ╚═╝╚══════╝           ║
║                                                                      ║
║        STUDENT GRADE MANAGEMENT SYSTEM  |  DSA Project               ║
║        VIT University  |  Python  |  Full Data Structures            ║
╚══════════════════════════════════════════════════════════════════════╝
        """)

    def main_menu(self):
        """Main menu loop."""
        while True:
            self._clear()
            self.banner()
            print("""
  ┌──────────────────────────────────────────────────┐
  │  MAIN MENU                                       │
  ├──────────────────────────────────────────────────┤
  │  [1]  Student Management (Add / Remove / Search) │
  │  [2]  Grade Management  (Add / View Grades)      │
  │  [3]  Rankings & Analytics                       │
  │  [4]  Department Reports                         │
  │  [5]  Course & Prerequisites (Graph DSA)         │
  │  [6]  Sorting Demo (Merge/Quick/Insertion)       │
  │  [7]  Search Demo (BST / Binary Search / Trie)   │
  │  [8]  Queue & Stack Demo                         │
  │  [9]  All Students List                          │
  │  [0]  Exit                                       │
  └──────────────────────────────────────────────────┘""")
            choice = input("\n  Enter choice: ").strip()
            if   choice == '1': self.student_menu()
            elif choice == '2': self.grade_menu()
            elif choice == '3': self.rankings_menu()
            elif choice == '4': self.dept_report_menu()
            elif choice == '5': self.course_graph_menu()
            elif choice == '6': self.sorting_demo()
            elif choice == '7': self.search_demo()
            elif choice == '8': self.queue_stack_demo()
            elif choice == '9': self.all_students()
            elif choice == '0':
                print("\n  Goodbye! Data Structures power the world 🚀\n")
                break
            else:
                print("  Invalid choice.")

    # ── Sub Menus ──────────────────────────────────────────────────────────

    def student_menu(self):
        while True:
            self._clear()
            print("""
  ┌────────────────────────────────────────┐
  │  STUDENT MANAGEMENT                    │
  ├────────────────────────────────────────┤
  │  [1] View Student Profile              │
  │  [2] Add New Student                   │
  │  [3] Remove Student                    │
  │  [4] Search by Name Prefix (Trie)      │
  │  [5] BST Range Query (Reg No range)    │
  │  [6] Undo Last Operation               │
  │  [0] Back                              │
  └────────────────────────────────────────┘""")
            c = input("\n  Choice: ").strip()
            if c == '0':
                break
            elif c == '1':
                reg = input("  Enter Register Number: ").strip().upper()
                s   = self.gms.get_student(reg)
                if s:
                    self.gms.print_student_card(s)
                else:
                    print(f"  Student {reg} not found.")
            elif c == '2':
                reg  = input("  Register No   : ").strip().upper()
                name = input("  Full Name     : ").strip()
                dept = input(f"  Department {self.gms.DEPARTMENTS}: ").strip()
                bat  = int(input("  Batch Year    : ").strip())
                ok, msg = self.gms.add_student(reg, name, dept, bat)
                print(f"  {msg}")
            elif c == '3':
                reg = input("  Register No to remove: ").strip().upper()
                ok, msg = self.gms.remove_student(reg)
                print(f"  {msg}")
            elif c == '4':
                prefix = input("  Enter name prefix: ").strip()
                results = self.gms.search_by_name_prefix(prefix)
                if results:
                    print(f"\n  Found {len(results)} student(s):")
                    for s in results:
                        print(f"    {s.reg_no:<14} {s.name:<28} CGPA: {s.cgpa}")
                else:
                    print("  No students found.")
            elif c == '5':
                low  = input("  Low  Reg No: ").strip().upper()
                high = input("  High Reg No: ").strip().upper()
                results = self.gms.bst.range_query(low, high)
                print(f"\n  {len(results)} students in range [{low} – {high}]:")
                for s in results:
                    print(f"    {s.reg_no:<14} {s.name:<28} CGPA: {s.cgpa}")
            elif c == '6':
                msg = self.gms.undo_last_operation()
                print(f"  {msg}")
            self._pause()

    def grade_menu(self):
        while True:
            self._clear()
            print("""
  ┌────────────────────────────────────────┐
  │  GRADE MANAGEMENT                      │
  ├────────────────────────────────────────┤
  │  [1] View Student Grades               │
  │  [2] Add Grade Manually                │
  │  [3] Enqueue Grade (Queue Demo)        │
  │  [4] Process Grade Queue               │
  │  [5] View Arrears                      │
  │  [0] Back                              │
  └────────────────────────────────────────┘""")
            c = input("\n  Choice: ").strip()
            if c == '0':
                break
            elif c == '1':
                reg = input("  Register No: ").strip().upper()
                s   = self.gms.get_student(reg)
                if s:
                    self.gms.print_student_card(s)
                else:
                    print("  Not found.")
            elif c == '2':
                reg  = input("  Register No    : ").strip().upper()
                code = input("  Subject Code   : ").strip().upper()
                gl   = input("  Grade (S/A+/A/B+/B/C/D/F): ").strip().upper()
                sem  = int(input("  Semester       : ").strip())
                ok, msg = self.gms.add_grade(reg, code, gl, sem)
                print(f"  {msg}")
            elif c == '3':
                reg  = input("  Register No    : ").strip().upper()
                code = input("  Subject Code   : ").strip().upper()
                gl   = input("  Grade          : ").strip().upper()
                sem  = int(input("  Semester       : ").strip())
                self.gms.enqueue_result_processing(reg, code, gl, sem)
                print(f"  ✅ Enqueued. Queue size: {self.gms.result_queue.size()}")
            elif c == '4':
                if self.gms.result_queue.is_empty():
                    print("  Queue is empty.")
                else:
                    msgs = self.gms.process_result_queue()
                    for m in msgs:
                        print(f"  {m}")
            elif c == '5':
                reg = input("  Register No: ").strip().upper()
                s   = self.gms.get_student(reg)
                if s:
                    arrears = s.get_arrears()
                    if arrears:
                        print(f"\n  {s.name} has {len(arrears)} arrear(s):")
                        for g in arrears:
                            print(f"    {g.subject_code} — {g.subject_name} (Sem {g.semester})")
                    else:
                        print(f"  {s.name} has no arrears. 🎉")
                else:
                    print("  Not found.")
            self._pause()

    def rankings_menu(self):
        self._clear()
        n = int(input("  Show top-N students (e.g. 5): ").strip() or "5")
        self.gms.print_rankings(n)
        print("\n  CGPA Range Query:")
        low  = float(input("  Min CGPA: ").strip() or "8.0")
        high = float(input("  Max CGPA: ").strip() or "10.0")
        results = self.gms.get_cgpa_range(low, high)
        print(f"\n  Students with CGPA between {low} and {high}:")
        for s in results:
            rank = self.gms.ranking_heap.get_rank(
                list(self.gms._students.values()), s.reg_no)
            print(f"    Rank #{rank:<4} {s.reg_no:<14} {s.name:<28} CGPA: {s.cgpa}")
        self._pause()

    def dept_report_menu(self):
        self._clear()
        print(f"  Departments: {', '.join(self.gms.DEPARTMENTS)}")
        dept = input("  Enter Department: ").strip().upper()
        self.gms.print_department_report(dept)
        self._pause()

    def course_graph_menu(self):
        self._clear()
        print("""
  Course Graph Options:
  [1] View Course Details & Prerequisites
  [2] Check Enrollment Eligibility
  [3] Topological Study Order
  [4] Detect Cycles in Curriculum
        """)
        c = input("  Choice: ").strip()
        if c == '1':
            code = input("  Course Code: ").strip().upper()
            self.gms.print_course_graph_info(code)
        elif c == '2':
            reg  = input("  Register No  : ").strip().upper()
            code = input("  Course Code  : ").strip().upper()
            ok, msg = self.gms.check_course_eligibility(reg, code)
            print(f"  {msg}")
        elif c == '3':
            order = self.gms.course_graph.topological_sort()
            print("\n  Recommended Study Order (Topological Sort):")
            for i, code in enumerate(order, 1):
                c_info = self.gms.course_registry.get(code)
                name   = c_info['name'] if c_info else code
                print(f"  {i:>3}. {code:<12} {name}")
        elif c == '4':
            has_cycle = self.gms.course_graph.has_cycle()
            status = "⚠️  Cycle detected!" if has_cycle else "✅ No cycles — valid DAG"
            print(f"\n  {status}")
        self._pause()

    def sorting_demo(self):
        self._clear()
        students = list(self.gms._students.values())
        print("  ── Merge Sort by CGPA (Descending) ──")
        ms = Sorter.merge_sort(students, key=lambda s: s.cgpa, reverse=True)
        for i, s in enumerate(ms[:8], 1):
            print(f"  {i:>2}. {s.reg_no:<14} {s.name:<28} CGPA: {s.cgpa}")

        print("\n  ── Quick Sort by Name (Ascending) ──")
        qs = Sorter.quick_sort(students, key=lambda s: s.name.lower())
        for s in qs[:8]:
            print(f"  {s.reg_no:<14} {s.name:<28} CGPA: {s.cgpa}")

        print("\n  ── Insertion Sort by Batch ──")
        ins = Sorter.insertion_sort(students, key=lambda s: s.batch)
        for s in ins[:8]:
            print(f"  Batch {s.batch}  {s.reg_no:<14} {s.name}")
        self._pause()

    def search_demo(self):
        self._clear()
        print("  ── BST Search Demo ──")
        reg = input("  Search by Register No (BST O(log n)): ").strip().upper()
        s   = self.gms.bst.search(reg)
        print(f"  {'Found: ' + str(s) if s else 'Not found.'}")

        print("\n  ── Binary Search on CGPA-sorted List ──")
        sorted_s = Sorter.merge_sort(list(self.gms._students.values()),
                                     key=lambda x: x.cgpa)
        target   = float(input("  Search for exact CGPA: ").strip() or "8.0")
        idx      = BinarySearch.search_by_cgpa(sorted_s, target)
        print(f"  {'Found at index ' + str(idx) + ': ' + str(sorted_s[idx]) if idx >= 0 else 'Not found.'}")

        print("\n  ── Trie Prefix Search ──")
        prefix = input("  Enter name prefix: ").strip()
        results = self.gms.search_by_name_prefix(prefix)
        print(f"  {len(results)} match(es): {[s.name for s in results]}")
        self._pause()

    def queue_stack_demo(self):
        self._clear()
        print("  ── STACK (Undo History) ──")
        history = self.gms.undo_stack.history()
        if history:
            print(f"  Stack size: {self.gms.undo_stack.size()}")
            for op, data, ts in history[:5]:
                print(f"   [{ts.strftime('%H:%M:%S')}] {op}")
        else:
            print("  Undo stack is empty. Perform some operations first.")

        print("\n  ── QUEUE (Pending Results) ──")
        print(f"  Queue size: {self.gms.result_queue.size()}")
        if not self.gms.result_queue.is_empty():
            for task in self.gms.result_queue:
                print(f"  → {task}")
        else:
            print("  Queue is empty.")
            print("  Enqueuing 2 demo tasks...")
            self.gms.enqueue_result_processing("22BCE0001","CSE3001","A",5)
            self.gms.enqueue_result_processing("22BCE0002","CSE3001","A+",5)
            print(f"  Queue size now: {self.gms.result_queue.size()}")
            print("  Processing queue...")
            msgs = self.gms.process_result_queue()
            for m in msgs:
                print(f"    {m}")
        self._pause()

    def all_students(self):
        self._clear()
        self.gms._header("All Students — Sorted by CGPA")
        students = self.gms.all_students_sorted("cgpa", reverse=True)
        print(f"  {'#':<4} {'Reg No':<14} {'Name':<28} {'Dept':<8} "
              f"{'Batch':<7} {'CGPA'}")
        self.gms._divider()
        for i, s in enumerate(students, 1):
            print(f"  {i:<4} {s.reg_no:<14} {s.name:<28} {s.department:<8} "
                  f"{s.batch:<7} {s.cgpa}")
        self.gms._divider("═")
        print(f"  Total students in system: {len(students)}")
        self._pause()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14: ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  🎓 Initializing VIT Grade Management System...")
    print("  Loading data structures: Linked List, BST, Trie, Heap, Graph...\n")
    app = CLI()
    app.main_menu()
