from typing import Any, cast
from graph1 import Graph
import pytest

def test_directed_add_and_degree():
    g = Graph[str](directed=True)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    assert set(g.vertices()) == {"A", "B", "C"}
    assert set(g.get_adjacent_vertices("A")) == {"B", "C"}
    assert g.has_edge("A", "B")
    assert not g.has_edge("B", "A")
    assert g.out_degree("A") == 2
    assert g.in_degree("B") == 1

def test_undirected_duplicates_avoided():
    g = Graph[int]()
    g.add_edge(1, 2)
    g.add_edge(1, 2)
    assert g.has_edge(1, 2) and g.has_edge(2, 1)
    edges = g.edges()
    assert len(edges) == 1
    assert set(edges[0]) == {1, 2}

def test_remove_edge_and_vertex():
    g = Graph[str]()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.remove_edge("A", "B")
    assert not g.has_edge("A", "B")
    g.remove_vertex("C")
    assert not g.has_vertex("C")
    assert not g.has_edge("B", "C")

def test_getter_immutability():
    g = Graph[str]()
    g.add_edge("A", "B")
    adj = g.get_adjacent_vertices("A")
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, adj).append("Z")
    assert g.has_edge("A", "B")
    assert not g.has_edge("A", "Z")
