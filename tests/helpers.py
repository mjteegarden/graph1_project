import json
from typing import Any, Dict, Tuple

def expect_raises(exc_types, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_types:
        return
    except Exception as ex:
        raise AssertionError(f"Expected {exc_types}, but got {type(ex).__name__}: {ex}") from ex
    else:
        raise AssertionError(f"Expected {exc_types}, but no exception was raised")

def _ser_vertex(v: Any) -> Any:
    if isinstance(v, tuple):
        return {"__type__": "tuple", "items": [_ser_vertex(x) for x in v]}
    return v

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
