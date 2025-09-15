
# graph1.py


from __future__ import annotations
from typing import Dict, Generic, Hashable, Iterable, Optional, Set, Tuple, Type, TypeVar, Callable
import re

T = TypeVar("T", bound=Hashable)

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")  # non-printable/control chars

def _safe_str(x: object, max_len: int = 120) -> str:
    """
    Best-effort safe-ish string for logs/printing:
    - uses str() then strips ANSI escapes and control chars
    - truncates long values to avoid log spam
    """
    s = str(x)
    s = _ANSI_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s

class Graph(Generic[T]):
    """
    Simple graph with defense-in-depth:
    - adjacency stored as sets to avoid duplicate edges
    - immutable snapshots from getters
    - input validation hooks for untrusted data
    - optional capacity caps to prevent accidental blow-ups
    """

    def __init__(
        self,
        directed: bool = False,
        *,
        allow_self_loops: bool = False,
        # Validation knobs (all optional; defaults are permissive but safe)
        restrict_vertex_types: Optional[Tuple[Type, ...]] = None,
        vertex_validator: Optional[Callable[[T], bool]] = None,
        # Capacity guards
        max_vertices: Optional[int] = None,
        max_degree: Optional[int] = None,
    ):
        self._g: Dict[T, Set[T]] = {}
        self.directed = directed
        self.allow_self_loops = allow_self_loops
        self._restrict_vertex_types = restrict_vertex_types
        self._vertex_validator = vertex_validator
        self._max_vertices = max_vertices
        self._max_degree = max_degree

    # ---------- internal helpers ----------

    def _validate_vertex_type(self, v: T) -> None:
        # Hashability is required (enforced by TypeVar at type-check time, but we also gate at runtime)
        try:
            hash(v)  # noqa: B023
        except Exception as e:
            raise TypeError(f"Vertex must be hashable; got {type(v).__name__}") from e

        # Optional hard type restrictions (e.g., only allow int/str/tuple)
        if self._restrict_vertex_types is not None and not isinstance(v, self._restrict_vertex_types):
            allowed = ", ".join(t.__name__ for t in self._restrict_vertex_types)
            raise TypeError(f"Vertex type {type(v).__name__} not allowed (allowed: {allowed})")

        # Optional custom validator (e.g., limit string length, regex, etc.)
        if self._vertex_validator is not None and not self._vertex_validator(v):
            raise ValueError("Vertex failed custom validation")

    def _check_capacity_before_add_vertex(self) -> None:
        if self._max_vertices is not None and len(self._g) >= self._max_vertices:
            raise OverflowError(f"Vertex cap exceeded (max_vertices={self._max_vertices})")

    def _check_degree_before_add_edge(self, src: T, dest: T) -> None:
        # Degree guard applies to outgoing degree (and symmetric for undirected)
        if self._max_degree is None:
            return
        # if edge already exists, no growth => skip
        if src in self._g and dest in self._g[src]:
            return
        if len(self._g.get(src, ())) >= self._max_degree:
            raise OverflowError(f"Degree cap exceeded for {src!r} (max_degree={self._max_degree})")
        if not self.directed:
            # undirected adds both ways (but we only store once per endpoint)
            if len(self._g.get(dest, ())) >= self._max_degree:
                raise OverflowError(f"Degree cap exceeded for {dest!r} (max_degree={self._max_degree})")

    # ---------- mutation ----------

    def add_vertex(self, v: T) -> None:
        self._validate_vertex_type(v)
        if v not in self._g:
            self._check_capacity_before_add_vertex()
            self._g[v] = set()

    def add_edge(self, src: T, dest: T) -> None:
        # validate first
        self._validate_vertex_type(src)
        self._validate_vertex_type(dest)

        # self-loop policy
        if not self.allow_self_loops and src == dest:
            raise ValueError("Self-loops are disabled (set allow_self_loops=True to permit)")

        # ensure vertices
        self.add_vertex(src)
        self.add_vertex(dest)

        # capacity guard (degree)
        self._check_degree_before_add_edge(src, dest)

        # add edge(s)
        self._g[src].add(dest)
        if not self.directed:
            self._g[dest].add(src)

    def remove_edge(self, src: T, dest: T) -> None:
        if src in self._g:
            self._g[src].discard(dest)
        if not self.directed and dest in self._g:
            self._g[dest].discard(src)

    def remove_vertex(self, v: T) -> None:
        if v not in self._g:
            return
        # remove inbound edges
        for u in list(self._g):
            if v in self._g[u]:
                self._g[u].discard(v)
        # remove the vertex
        del self._g[v]

    # ---------- queries ----------

    def has_vertex(self, v: T) -> bool:
        return v in self._g

    def has_edge(self, src: T, dest: T) -> bool:
        return src in self._g and dest in self._g[src]

    def get_adjacent_vertices(self, v: T) -> Tuple[T, ...]:
        return tuple(self._g.get(v, ()))

    def vertices(self) -> Tuple[T, ...]:
        return tuple(self._g.keys())

    def edges(self) -> Tuple[Tuple[T, T], ...]:
        if self.directed:
            return tuple((u, v) for u, nbrs in self._g.items() for v in nbrs)
        # undirected: emit each edge once
        seen = set()
        out = []
        for u, nbrs in self._g.items():
            for v in nbrs:
                a, b = (u, v) if repr(u) < repr(v) else (v, u)
                if (a, b) not in seen:
                    seen.add((a, b))
                    out.append((a, b))
        return tuple(out)

    # degrees
    def out_degree(self, v: T) -> int:
        return len(self._g.get(v, ()))

    def in_degree(self, v: T) -> int:
        if not self.directed:
            return len(self._g.get(v, ()))
        return sum(1 for u in self._g if v in self._g[u])

    def degree(self, v: T) -> int:
        return self.out_degree(v) if self.directed else self.out_degree(v)

    # ---------- dunder ----------

    def __len__(self) -> int:
        return len(self._g)

    def __contains__(self, v: T) -> bool:
        return v in self._g

    def __str__(self) -> str:
        kind = "Directed" if self.directed else "Undirected"
        lines = []
        for u in self.vertices():
            nbrs = ", ".join(_safe_str(x) for x in sorted(self._g[u], key=_safe_str))
            lines.append(f"{_safe_str(u)}: [{nbrs}]")
        return f"{kind}Graph {{\n  " + "\n  ".join(lines) + "\n}}"

