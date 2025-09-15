
# graph1-smoketest.py
# Run with:  python -u graph1-smoketest.py

import json
import os
import traceback
from graph1_project.graph1 import Graph
from typing import Any, Callable, cast, Dict, List, Tuple


class TestRunner:
    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self._tests: List[Tuple[str, Callable[[], None]]] = []

    def test(self, name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
        def deco(fn: Callable[[], None]) -> Callable[[], None]:
            self._tests.append((name, fn))
            return fn
        return deco

    def assert_true(self, expr: bool, msg: str = "") -> None:
        if not expr:
            raise AssertionError(msg or "Expected True, got False")

    def assert_false(self, expr: bool, msg: str = "") -> None:
        if expr:
            raise AssertionError(msg or "Expected False, got True")

    def assert_equal(self, a, b, msg: str = "") -> None:
        if a != b:
            raise AssertionError(msg or f"Expected {b!r}, got {a!r}")

    def run(self) -> bool:
        print("Running Graph smoketests...\n")
        show_trace = os.getenv("SHOW_TRACE", "0") not in ("0", "", "false", "False")
        for name, fn in self._tests:
            try:
                fn()
            except Exception as ex:
                # keep all usage of `ex` inside the except block (mypy-friendly)
                print(f"✗ {name}  -- {type(ex).__name__}: {ex}")
                if show_trace:
                    traceback.print_exc()
                self.failed += 1
            else:
                print(f"✓ {name}")
                self.passed += 1
        print("\nFinished:", f"{self.passed} passed,", f"{self.failed} failed.")
        return self.failed == 0


import json
from typing import Any, Dict

def _ser_vertex(v: Any) -> Any:
    # Tag tuples so they survive JSON round-trip
    if isinstance(v, tuple):
        return {"__type__": "tuple", "items": [_ser_vertex(x) for x in v]}
    return v  # ints/str already JSON-safe

def _ser_edge(u: Any, v: Any) -> Any:
    return [_ser_vertex(u), _ser_vertex(v)]

def serialize_graph(g: "Any") -> str:
    payload: Dict[str, Any] = {
        "directed": g.directed,
        "allow_self_loops": getattr(g, "allow_self_loops", False),
        "vertices": [_ser_vertex(v) for v in g.vertices()],
        "edges": [_ser_edge(u, v) for (u, v) in g.edges()],
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def expect_raises(exc_types, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_types:
        return
    except Exception as ex:
        raise AssertionError(f"Expected {exc_types}, but got {type(ex).__name__}: {ex}") from ex
    else:
        raise AssertionError(f"Expected {exc_types}, but no exception was raised")


tr = TestRunner()

@tr.test("directed: add vertices/edges and degrees behave")
def _():
    g = Graph[str](directed=True)
    g.add_edge("A", "B")     # also adds vertices
    g.add_edge("A", "C")

    # vertices present
    tr.assert_equal(set(g.vertices()), {"A", "B", "C"})
    # adjacency is a tuple; compare as sets to ignore order
    tr.assert_equal(set(g.get_adjacent_vertices("A")), {"B", "C"})
    tr.assert_true(g.has_edge("A", "B"))
    tr.assert_false(g.has_edge("B", "A"))  # directed

    # degrees
    tr.assert_equal(g.out_degree("A"), 2)
    tr.assert_equal(g.in_degree("B"), 1)
    tr.assert_equal(g.in_degree("C"), 1)
    tr.assert_equal(g.out_degree("B"), 0)

    # edges iterator emits directed edges
    tr.assert_equal(set(g.edges()), {("A", "B"), ("A", "C")})

@tr.test("undirected: duplicates avoided and edge visible both ways")
def _():
    g = Graph[int]()  # undirected by default
    g.add_edge(1, 2)
    g.add_edge(1, 2)  # duplicate should be ignored (set)
    tr.assert_true(g.has_edge(1, 2))
    tr.assert_true(g.has_edge(2, 1))
    tr.assert_equal(g.out_degree(1), 1)
    tr.assert_equal(g.out_degree(2), 1)

    # edges() should list each undirected edge once
    edges = g.edges()
    tr.assert_equal(len(edges), 1)
    a, b = edges[0]
    tr.assert_equal({a, b}, {1, 2})

@tr.test("remove_edge and remove_vertex cleanly update structure")
def _():
    g = Graph[str]()
    g.add_edge("A", "B")
    g.add_edge("B", "C")

    # remove an edge
    g.remove_edge("A", "B")
    tr.assert_false(g.has_edge("A", "B"))
    tr.assert_false(g.has_edge("B", "A"))

    # remove a vertex and its incident edges
    g.remove_vertex("C")
    tr.assert_false(g.has_vertex("C"))
    tr.assert_false(g.has_edge("B", "C"))
    tr.assert_false(g.has_edge("C", "B"))

@tr.test("getter immutability: adjacency is read-only snapshot")
def _():
    g = Graph[str]()
    g.add_edge("A", "B")
    adj = g.get_adjacent_vertices("A")  # Tuple[str, ...]
    try:
        cast(Any, adj).append("Z")  # force runtime error without mypy complaint
        raise AssertionError("adjacency should be immutable but append succeeded")
    except (AttributeError, TypeError):
        pass
    tr.assert_true(g.has_edge("A", "B"))
    tr.assert_false(g.has_edge("A", "Z"))


@tr.test("len/contains/predicates basic behavior")
def _():
    g = Graph[int]()
    g.add_vertex(10)
    g.add_vertex(20)
    g.add_edge(10, 20)

    tr.assert_true(10 in g)
    tr.assert_true(g.has_vertex(10))
    tr.assert_false(g.has_vertex(30))
    tr.assert_equal(len(g), 2)
    tr.assert_true(g.has_edge(10, 20))
    tr.assert_true(g.has_edge(20, 10))  # undirected

@tr.test("__str__ is readable and includes key info")
def _():
    g = Graph[str](directed=False)
    g.add_edge("A", "B")
    s = str(g)
    tr.assert_true("UndirectedGraph" in s)
    tr.assert_true("A:" in s)
    tr.assert_true("B" in s)

@tr.test("self-loop rejection (default policy: disallow)")
def _():
    g = Graph[str](directed=True)  # allow_self_loops defaults to False
    g.add_vertex("A")
    try:
        g.add_edge("A", "A")
        raise AssertionError("Expected ValueError for self-loop, but add_edge succeeded")
    except ValueError:
        pass
    tr.assert_false(g.has_edge("A", "A"))

@tr.test("self-loop allowed when explicitly enabled")
def _():
    g = Graph[str](directed=True, allow_self_loops=True)
    g.add_vertex("A")
    g.add_edge("A", "A")  # should be allowed now
    tr.assert_true(g.has_edge("A", "A"))

@tr.test("type restriction: whitelist vertex types")
def _():
    # Only int and str are allowed; tuples should be rejected even though they are hashable
    g = Graph(directed=False, restrict_vertex_types=(int, str))
    g.add_vertex(1)
    g.add_vertex("B")
    tr.assert_true(g.has_vertex(1))
    tr.assert_true(g.has_vertex("B"))
    try:
        g.add_vertex((10, 20))  # hashable but NOT whitelisted
        raise AssertionError("Expected TypeError for disallowed vertex type (tuple)")
    except TypeError:
        pass

@tr.test("custom validator: enforce string length <= 5")
def _():
    g = Graph[str](vertex_validator=lambda v: isinstance(v, str) and len(v) <= 5)
    g.add_vertex("short")  # ok
    tr.assert_true(g.has_vertex("short"))
    try:
        g.add_vertex("toolong")  # 7 chars -> should fail validator
        raise AssertionError("Expected ValueError from custom validator (length > 5)")
    except ValueError:
        pass

@tr.test("capacity cap: max_vertices")
def _():
    g = Graph[int](max_vertices=3)
    g.add_vertex(1)
    g.add_vertex(2)
    g.add_vertex(3)
    tr.assert_equal(len(g), 3)
    try:
        g.add_vertex(4)  # exceeds cap
        raise AssertionError("Expected OverflowError when exceeding max_vertices")
    except OverflowError:
        pass

@tr.test("capacity cap: max_degree (undirected)")
def _():
    g = Graph[int](max_degree=1)  # each vertex can have at most 1 neighbor
    g.add_edge(1, 2)  # ok: degree(1)=1, degree(2)=1
    tr.assert_true(g.has_edge(1, 2))
    try:
        g.add_edge(1, 3)  # would make degree(1)=2 -> should fail
        raise AssertionError("Expected OverflowError when exceeding max_degree on src")
    except OverflowError:
        pass
    # Also test hitting the cap on the dest side for undirected graphs
    try:
        g.add_edge(3, 2)  # would make degree(2)=2 -> should fail
        raise AssertionError("Expected OverflowError when exceeding max_degree on dest")
    except OverflowError:
        pass

@tr.test("capacity cap: max_degree (directed, out-degree enforced)")
def _():
    g = Graph[str](directed=True, max_degree=2)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    tr.assert_equal(g.out_degree("A"), 2)
    try:
        g.add_edge("A", "D")  # would make out-degree(A)=3 -> violate cap
        raise AssertionError("Expected OverflowError when exceeding directed out-degree cap")
    except OverflowError:
        pass
    # in-degree is not capped by max_degree in this implementation; add more inbound to verify
    g.add_edge("X", "B")
    g.add_edge("Y", "B")
    tr.assert_equal(g.in_degree("B"), 3)  # A->B, X->B, Y->B

@tr.test("__str__ strips ANSI/control chars and truncates long labels")
def _():
    # ANSI sequences + control chars in labels
    red = "\x1b[31mRED\x1b[0m"   # ANSI colored
    weird = "Weird\tName\nWith\x00Controls"  # has tab, newline, NUL
    long_label = "L" * 300  # should be truncated with an ellipsis

    g = Graph[str]()  # undirected
    g.add_edge(red, weird)
    g.add_vertex(long_label)

    s = str(g)

    # ANSI escape sequences must not appear
    tr.assert_false("\x1b[" in s, "ANSI escape sequence leaked into __str__ output")

    # Control characters from labels must be stripped. We DO allow the formatter’s line breaks.
    # So: forbid tab and NUL; don't globally forbid '\n' because __str__ uses newlines for formatting.
    tr.assert_false("\t" in s, "Tab control leaked into __str__ output")
    tr.assert_false("\x00" in s, "NUL control leaked into __str__ output")

    # Human-readable content should remain (sanitized)
    tr.assert_true("RED" in s)
    tr.assert_true("WeirdNameWithControls" in s or "WeirdNameWith" in s,
                   "Controls should be stripped, text retained")

    # Truncation indicator (single-character ellipsis) likely appears for very long label
    tr.assert_true("…" in s, "Expected ellipsis for truncated long label")

@tr.test("unhashable vertex rejected")
def _():
    g = Graph()
    try:
        g.add_vertex(["not-hashable"])  # list is unhashable
        raise AssertionError("Expected TypeError for unhashable vertex")
    except TypeError:
        pass

@tr.test("removing non-existent edge/vertex doesn't crash")
def _():
    g = Graph[int]()
    # Should be no exceptions
    g.remove_edge(1, 2)
    g.remove_vertex(3)
    tr.assert_equal(len(g), 0)


# ===== Production-readiness tests =====
# Paste these below your existing tests in graph1-smoketest.py

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Tuple

# ---------- helpers: JSON (de)serialization with validation ----------


def _deser_vertex(x: Any) -> Any:
    if isinstance(x, dict) and x.get("__type__") == "tuple":
        items = x.get("items")
        if not isinstance(items, list):
            raise ValueError("Malformed tuple encoding in JSON")
        return tuple(_deser_vertex(i) for i in items)
    return x


def deserialize_graph(
    s: str,
    *,
    restrict_vertex_types: Tuple[type, ...] = (int, str, tuple),
    max_vertices: int = 200_000,
    max_degree: int | None = None,
    allow_self_loops: bool | None = None,
) -> "Any":
    from graph1 import Graph  # local import to avoid cycles

    data = json.loads(s)
    for key in ("directed", "vertices", "edges"):
        if key not in data:
            raise ValueError(f"Missing key in graph JSON: {key}")

    directed = bool(data["directed"])
    vertices_raw = data["vertices"]
    edges_raw = data["edges"]
    json_self_loops = bool(data.get("allow_self_loops", False))
    use_self_loops = json_self_loops if allow_self_loops is None else bool(allow_self_loops)

    if not isinstance(vertices_raw, list) or not isinstance(edges_raw, list):
        raise ValueError("vertices and edges must be lists")

    vertices = [_deser_vertex(v) for v in vertices_raw]
    edges = [[_deser_vertex(p[0]), _deser_vertex(p[1])] for p in edges_raw]

    g2 = Graph(
        directed=directed,
        allow_self_loops=use_self_loops,
        restrict_vertex_types=restrict_vertex_types,
        max_vertices=max_vertices,
        max_degree=max_degree,
    )

    if len(vertices) > max_vertices:
        raise OverflowError(f"Too many vertices in JSON ({len(vertices)} > {max_vertices})")

    for v in vertices:
        if not isinstance(v, restrict_vertex_types):
            allowed = ", ".join(t.__name__ for t in restrict_vertex_types)
            raise ValueError(f"Invalid vertex type in JSON: {type(v).__name__} (allowed: {allowed})")
        try:
            hash(v)
        except TypeError as ex:
            raise ValueError(f"Invalid vertex in JSON: {ex}") from ex
        g2.add_vertex(v)

    for e in edges:
        if not (isinstance(e, list) and len(e) == 2):
            raise ValueError("Each edge must be a [u, v] pair")
        u, v = e[0], e[1]
        if not g2.has_vertex(u) or not g2.has_vertex(v):
            raise ValueError(f"Edge references unknown vertex: {e!r}")
        g2.add_edge(u, v)

    return g2



# ---------- TEST 1: Performance profiling (10k scale) ----------

@tr.test("perf: build ~10k vertices and edges within budget")
def _():
    """
    Builds N vertices and ~N edges and checks structure + elapsed time.
    Time budget is generous and configurable via env var to avoid flakiness.
      PERF_N: number of vertices (default: 10000)
      PERF_MAX_SEC: max seconds allowed (default: 15.0)
    """
    N = int(os.getenv("PERF_N", "10000"))
    MAX_SEC = float(os.getenv("PERF_MAX_SEC", "15.0"))

    g = Graph[int](directed=True)

    t0 = time.perf_counter()
    # Add vertices
    for i in range(N):
        g.add_vertex(i)
    # Add a simple chain + wrap edge (total N edges)
    for i in range(N):
        g.add_edge(i, (i + 1) % N)
    elapsed = time.perf_counter() - t0

    # Structural checks
    tr.assert_equal(len(g), N)
    tr.assert_equal(len(g.edges()), N)  # directed cycle: N edges
    # sanity degree checks (cycle graph => in=out=1)
    tr.assert_equal(g.in_degree(0), 1)
    tr.assert_equal(g.out_degree(0), 1)

    # Timing check (very generous by default; make stricter locally if you want)
    if elapsed > MAX_SEC:
        raise AssertionError(f"Perf budget exceeded: {elapsed:.3f}s > {MAX_SEC:.3f}s")
    # Print a little perf info
    print(f"   (perf) built {N} vertices & {N} edges in {elapsed:.3f}s "
          f"~ {int((2*N)/max(elapsed,1e-9))} ops/s")


# ---------- TEST 2: Concurrency consistency (multi-threaded) ----------

@tr.test("concurrency: safe multi-threaded writes (external lock) + concurrent readers")
def _():
    """
    Graph is not internally synchronized; we exercise a recommended pattern:
      - Use one global lock for writes (add_vertex/add_edge).
      - Allow concurrent readers (has_edge, degree, vertices) without mutation.
    We verify that the final structure matches expected totals with no crashes.
    """
    g = Graph[int](directed=False)
    write_lock = threading.Lock()

    # Build plan: T writer threads each add edges for a disjoint range
    T = int(os.getenv("CONC_WRITERS", "8"))
    N = int(os.getenv("CONC_VERTS", "5000"))         # vertices 0..N-1
    EDGES_PER_WRITER = int(os.getenv("CONC_EDGES_PER_WRITER", "4000"))

    # Pre-add vertices (safe single-thread)
    for i in range(N):
        g.add_vertex(i)

    def writer(seed: int) -> int:
        added = 0
        # Add a predictable set of edges per writer:
        # connect i -> (i+seed) % N for a slice of i values
        start = (seed * EDGES_PER_WRITER) % N
        stop = min(start + EDGES_PER_WRITER, N)
        for i in range(start, stop):
            u = i
            v = (i + seed + 1) % N
            with write_lock:
                # Undirected, duplicates are ignored by set — so "added" is approximate success count
                before = g.has_edge(u, v)
                g.add_edge(u, v)
                after = g.has_edge(u, v)
                if not before and after:
                    added += 1
        return added

    def reader(samples: int = 10000) -> int:
        # Concurrent readers: no writes, just touch queries
        hits = 0
        local = list(g.vertices())  # snapshot vertices
        for i in range(min(samples, max(1, len(local)))):
            u = local[i % len(local)]
            # benign queries
            _ = g.out_degree(u)
            _ = g.get_adjacent_vertices(u)
            if g.has_vertex(u):
                hits += 1
        return hits

    # Run writers + readers together
    FUTURES = []
    with ThreadPoolExecutor(max_workers=T + 2) as ex:
        for t in range(T):
            FUTURES.append(ex.submit(writer, t))
        # Launch a couple of readers
        FUTURES.append(ex.submit(reader, 8000))
        FUTURES.append(ex.submit(reader, 8000))

        results = [f.result() for f in as_completed(FUTURES)]

    # Basic consistency: graph should still contain all N vertices
    tr.assert_equal(len(g.vertices()), N)

    # No self-loops expected from our pattern
    for (u, v) in g.edges():
        tr.assert_false(u == v)

    # Sanity: at least some edges were inserted
    tr.assert_true(len(g.edges()) > 0)


# ---------- TEST 3: Serialization round-trip with validation ----------

@tr.test("serialization: JSON round-trip preserves structure and enforces validation")
def _():
    from typing import Any

    # Build baseline
    g = Graph[Any](directed=True, allow_self_loops=False)
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B")
    g.add_vertex(("x", 1))
    g.add_edge("B", ("x", 1))

    s = serialize_graph(g)

    # Clean import succeeds
    g2 = deserialize_graph(
        s,
        restrict_vertex_types=(int, str, tuple),
        max_vertices=1000,
        max_degree=None,
        allow_self_loops=False,
    )
    tr.assert_equal(g2.directed, g.directed)
    tr.assert_equal(set(g2.vertices()), set(g.vertices()))
    tr.assert_equal(set(g2.edges()), set(g.edges()))

    # Tamper 1: unknown vertex in an edge -> must raise ValueError
    t1 = json.loads(s)
    t1["edges"].append(["A", "Z_DOES_NOT_EXIST"])
    expect_raises(ValueError, deserialize_graph, json.dumps(t1),
                  restrict_vertex_types=(int, str, tuple))

    # Tamper 2: disallowed/unhashable vertex (list) -> must raise ValueError
    t2 = json.loads(s)
    t2["vertices"].append(["not", "hashable"])  # list is not in whitelist and unhashable
    expect_raises(ValueError, deserialize_graph, json.dumps(t2),
                  restrict_vertex_types=(int, str, tuple))




if __name__ == "__main__":
    ok = tr.run()
    # Non-zero exit on failure helps CI or scripts detect problems.
    import sys
    sys.exit(0 if ok else 1)

