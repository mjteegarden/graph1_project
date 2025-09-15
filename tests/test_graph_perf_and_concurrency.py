
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import pytest
from graph1 import Graph
from test_helpers import serialize_graph, deserialize_graph, expect_raises
import json

@pytest.mark.perf
def test_perf_10k():
    N = int(os.getenv("PERF_N", "10000"))
    MAX_SEC = float(os.getenv("PERF_MAX_SEC", "15.0"))
    g = Graph[int](directed=True)

    t0 = time.perf_counter()
    for i in range(N):
        g.add_vertex(i)
    for i in range(N):
        g.add_edge(i, (i + 1) % N)
    elapsed = time.perf_counter() - t0

    assert len(g) == N
    assert len(g.edges()) == N
    assert g.in_degree(0) == 1 and g.out_degree(0) == 1
    assert elapsed <= MAX_SEC, f"built {N}+{N} in {elapsed:.3f}s > {MAX_SEC:.3f}s"

@pytest.mark.concurrency
def test_concurrency_writers_with_lock_and_readers():
    g = Graph[int](directed=False)
    from threading import Lock
    lock = Lock()

    T = int(os.getenv("CONC_WRITERS", "8"))
    N = int(os.getenv("CONC_VERTS", "5000"))
    EDGES_PER_WRITER = int(os.getenv("CONC_EDGES_PER_WRITER", "4000"))

    for i in range(N):
        g.add_vertex(i)

    def writer(seed: int) -> int:
        added = 0
        start = (seed * EDGES_PER_WRITER) % N
        stop  = min(start + EDGES_PER_WRITER, N)
        for i in range(start, stop):
            u, v = i, (i + seed + 1) % N
            with lock:
                before = g.has_edge(u, v)
                g.add_edge(u, v)
                after = g.has_edge(u, v)
                if not before and after:
                    added += 1
        return added

    def reader(samples: int = 8000) -> int:
        hits = 0
        verts = list(g.vertices())
        if not verts:
            return 0
        L = len(verts)
        for i in range(samples):
            u = verts[i % L]
            _ = g.out_degree(u)
            _ = g.get_adjacent_vertices(u)
            if g.has_vertex(u):
                hits += 1
        return hits

    futures = []
    with ThreadPoolExecutor(max_workers=T + 2) as ex:
        for t in range(T):
            futures.append(ex.submit(writer, t))
        futures.append(ex.submit(reader, 8000))
        futures.append(ex.submit(reader, 8000))
        _ = [f.result() for f in as_completed(futures)]

    assert len(g.vertices()) == N
    assert len(g.edges()) > 0
    for (u, v) in g.edges():
        assert u != v

def test_serialization_roundtrip_and_validation():
    g = Graph[Any](directed=True, allow_self_loops=False)
    g.add_vertex("A"); g.add_vertex("B"); g.add_edge("A", "B")
    g.add_vertex(("x", 1)); g.add_edge("B", ("x", 1))

    s = serialize_graph(g)
    g2 = deserialize_graph(s, restrict_vertex_types=(int, str, tuple),
                           max_vertices=1000, max_degree=None, allow_self_loops=False)
    assert g2.directed == g.directed
    assert set(g2.vertices()) == set(g.vertices())
    assert set(g2.edges()) == set(g.edges())

    t1 = json.loads(s)
    t1["edges"].append(["A", "Z_DOES_NOT_EXIST"])
    expect_raises(ValueError, deserialize_graph, json.dumps(t1),
                  restrict_vertex_types=(int, str, tuple))

    t2 = json.loads(s)
    t2["vertices"].append(["not","hashable"])
    expect_raises(ValueError, deserialize_graph, json.dumps(t2),
                  restrict_vertex_types=(int, str, tuple))
