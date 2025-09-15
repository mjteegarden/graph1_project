
from graph1 import Graph
import pytest

def test_self_loop_rejection_default():
    g = Graph[str](directed=True)
    g.add_vertex("A")
    with pytest.raises(ValueError):
        g.add_edge("A", "A")
    assert not g.has_edge("A", "A")

def test_self_loop_allow_when_enabled():
    g = Graph[str](directed=True, allow_self_loops=True)
    g.add_vertex("A")
    g.add_edge("A", "A")
    assert g.has_edge("A", "A")

def test_type_whitelist():
    g = Graph(directed=False, restrict_vertex_types=(int, str))
    g.add_vertex(1)
    g.add_vertex("B")
    with pytest.raises(TypeError):
        g.add_vertex((10, 20))

def test_custom_validator():
    g = Graph[str](vertex_validator=lambda v: isinstance(v, str) and len(v) <= 5)
    g.add_vertex("short")
    with pytest.raises(ValueError):
        g.add_vertex("toolong")

def test_caps():
    g = Graph[int](max_vertices=3)
    g.add_vertex(1); g.add_vertex(2); g.add_vertex(3)
    assert len(g) == 3
    with pytest.raises(OverflowError):
        g.add_vertex(4)

def test_caps_degree_undirected():
    g = Graph[int](max_degree=1)
    g.add_edge(1, 2)
    with pytest.raises(OverflowError):
        g.add_edge(1, 3)
    with pytest.raises(OverflowError):
        g.add_edge(3, 2)
