"""Skill taxonomy: alias resolution and hierarchical inheritance.

This module is our lexical answer to two of the research gaps the study
identifies:

* **Vocabulary mismatch** - aliases collapse "js", "ES6" and "ECMAScript" onto
  one canonical ``javascript``, so a candidate is not penalised for word choice.
* **Knowledge sparsity** (Paper 2's motivation for a knowledge graph) - parent
  edges encode implication, so shipping Django counts as evidence of Python
  without the candidate having to spell it out.

The expansion is deliberately **one-directional**: a skill implies its
ancestors, never its descendants. Someone who lists Django knows Python; someone
who lists Python has *not* thereby shown they know Django. Getting this backwards
would let a generalist match every specialist requirement, which is exactly the
kind of false positive that makes automated screening untrustworthy.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import yaml

from .config import ONTOLOGY_PATH

_WHITESPACE = re.compile(r"\s+")


class OntologyError(ValueError):
    """Raised when the ontology file is malformed."""


def normalise(surface: str) -> str:
    """Fold a raw surface form to its lookup key."""
    return _WHITESPACE.sub(" ", surface.strip().lower())


@dataclass(frozen=True)
class SkillNode:
    name: str
    category: str = "other"
    aliases: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()


@dataclass
class SkillOntology:
    """An immutable-in-practice skill graph with alias and ancestor lookup."""

    nodes: dict[str, SkillNode]
    #: normalised surface form -> canonical skill name
    _alias_index: dict[str, str] = field(default_factory=dict, repr=False)
    #: canonical name -> direct children
    _children: dict[str, set[str]] = field(default_factory=dict, repr=False)
    #: memoised transitive ancestor sets
    _ancestor_cache: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SkillOntology":
        path = Path(path) if path is not None else ONTOLOGY_PATH
        if not path.exists():
            raise OntologyError(f"ontology file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = raw.get("skills") or {}
        if not entries:
            raise OntologyError(f"ontology file {path} defines no skills")

        nodes: dict[str, SkillNode] = {}
        for name, body in entries.items():
            body = body or {}
            canonical = normalise(str(name))
            nodes[canonical] = SkillNode(
                name=canonical,
                category=str(body.get("category", "other")),
                aliases=tuple(normalise(str(a)) for a in body.get("aliases", ())),
                parents=tuple(normalise(str(p)) for p in body.get("parents", ())),
            )

        ontology = cls(nodes=nodes)
        ontology._build_indexes()
        return ontology

    def _build_indexes(self) -> None:
        self._alias_index = {}
        self._children = {name: set() for name in self.nodes}

        for node in self.nodes.values():
            # A canonical name is always its own surface form.
            self._alias_index[node.name] = node.name

            # Canonical names are hyphenated, but resumes write them with
            # spaces ("machine learning", "GitHub Actions"). Register the
            # spaced spelling too, or the phrase matcher tokenises the two
            # forms differently and never fires. Registered first so an
            # explicit alias on another skill can still override it below.
            spaced = node.name.replace("-", " ")
            if spaced != node.name:
                self._alias_index.setdefault(spaced, node.name)

            for alias in node.aliases:
                clash = self._alias_index.get(alias)
                if clash is not None and clash != node.name:
                    raise OntologyError(
                        f"alias {alias!r} is claimed by both {clash!r} and {node.name!r}"
                    )
                self._alias_index[alias] = node.name

            for parent in node.parents:
                if parent not in self.nodes:
                    raise OntologyError(
                        f"skill {node.name!r} lists unknown parent {parent!r}"
                    )
                self._children[parent].add(node.name)

        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        """Depth-first cycle check over parent edges.

        A cycle would make ``ancestors`` loop forever, so this fails loudly at
        load time rather than hanging during a screening run.
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(self.nodes, WHITE)

        def visit(name: str, trail: list[str]) -> None:
            colour[name] = GREY
            for parent in self.nodes[name].parents:
                if colour[parent] == GREY:
                    cycle = " -> ".join(trail + [name, parent])
                    raise OntologyError(f"cycle in skill parents: {cycle}")
                if colour[parent] == WHITE:
                    visit(parent, trail + [name])
            colour[name] = BLACK

        for name in self.nodes:
            if colour[name] == WHITE:
                visit(name, [])

    # -------------------------------------------------------------- lookups

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and normalise(name) in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def skill_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.nodes))

    def canonical(self, surface: str) -> str | None:
        """Resolve any surface form to its canonical skill name, or None."""
        return self._alias_index.get(normalise(surface))

    def surface_forms(self) -> Iterator[tuple[str, str]]:
        """Yield ``(surface, canonical)`` for every known spelling.

        Feeds the spaCy ``PhraseMatcher`` in :mod:`talentlens.extraction`.
        """
        for surface, canonical in self._alias_index.items():
            yield surface, canonical

    def category(self, skill: str) -> str:
        node = self.nodes.get(normalise(skill))
        return node.category if node else "other"

    # ------------------------------------------------------------ traversal

    def ancestors(self, skill: str) -> frozenset[str]:
        """All skills transitively implied by holding ``skill`` (excluding it)."""
        name = normalise(skill)
        if name not in self.nodes:
            return frozenset()
        cached = self._ancestor_cache.get(name)
        if cached is not None:
            return cached

        seen: set[str] = set()
        queue = deque(self.nodes[name].parents)
        while queue:
            current = queue.popleft()
            if current in seen or current not in self.nodes:
                continue
            seen.add(current)
            queue.extend(self.nodes[current].parents)

        result = frozenset(seen)
        self._ancestor_cache[name] = result
        return result

    def children(self, skill: str) -> frozenset[str]:
        return frozenset(self._children.get(normalise(skill), ()))

    def descendants(self, skill: str) -> frozenset[str]:
        """Every skill that transitively implies ``skill``.

        The inverse of :meth:`ancestors`, and the direction evidence flows in:
        a narrative sentence mentioning Django is evidence for a Python claim,
        because Django implies Python. Used by :mod:`talentlens.grounding` to
        decide whether a claimed skill is demonstrated anywhere in the text.
        """
        name = normalise(skill)
        if name not in self.nodes:
            return frozenset()
        seen: set[str] = set()
        queue = deque(self._children.get(name, ()))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self._children.get(current, ()))
        return frozenset(seen)

    def siblings(self, skill: str) -> frozenset[str]:
        """Skills sharing at least one direct parent with ``skill``."""
        name = normalise(skill)
        node = self.nodes.get(name)
        if node is None:
            return frozenset()
        out: set[str] = set()
        for parent in node.parents:
            out |= self._children.get(parent, set())
        out.discard(name)
        return frozenset(out)

    def expand(self, skills: Iterable[str]) -> set[str]:
        """Return ``skills`` plus every ancestor they imply.

        Unknown skills are preserved verbatim rather than dropped, so a resume
        listing a niche tool we have not catalogued still contributes to Dice
        overlap when the JD names that same tool.
        """
        out: set[str] = set()
        for raw in skills:
            canonical = self.canonical(raw) or normalise(raw)
            if not canonical:
                continue
            out.add(canonical)
            out |= self.ancestors(canonical)
        return out

    def resolve_all(self, skills: Iterable[str]) -> set[str]:
        """Canonicalise a set of surface forms without expanding ancestors."""
        out: set[str] = set()
        for raw in skills:
            canonical = self.canonical(raw) or normalise(raw)
            if canonical:
                out.add(canonical)
        return out

    # ------------------------------------------------- transferability search

    def distance(self, a: str, b: str, max_hops: int = 3) -> int | None:
        """Undirected hop count between two skills, or None if farther apart.

        Traversal is undirected here (unlike :meth:`expand`) because this
        measures *conceptual proximity for retraining purposes*, not evidence of
        possession. Flask and Django are one hop apart through Python, which is
        the relationship a recruiter cares about when judging how quickly a
        candidate could pick the other one up.
        """
        start, goal = normalise(a), normalise(b)
        if start not in self.nodes or goal not in self.nodes:
            return None
        if start == goal:
            return 0

        seen = {start}
        frontier = deque([(start, 0)])
        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            neighbours = set(self.nodes[current].parents) | self._children.get(
                current, set()
            )
            for neighbour in neighbours:
                if neighbour == goal:
                    return depth + 1
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append((neighbour, depth + 1))
        return None

    def shortest_path(self, a: str, b: str, max_hops: int = 3) -> tuple[str, ...]:
        """Undirected shortest path between two skills, empty if unreachable."""
        start, goal = normalise(a), normalise(b)
        if start not in self.nodes or goal not in self.nodes:
            return ()
        if start == goal:
            return (start,)

        seen = {start}
        frontier: deque[tuple[str, tuple[str, ...]]] = deque([(start, (start,))])
        while frontier:
            current, path = frontier.popleft()
            if len(path) > max_hops:
                continue
            neighbours = set(self.nodes[current].parents) | self._children.get(
                current, set()
            )
            for neighbour in sorted(neighbours):
                if neighbour == goal:
                    return path + (goal,)
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append((neighbour, path + (neighbour,)))
        return ()

    def nearest_held(
        self, target: str, held: Iterable[str], max_hops: int = 2
    ) -> tuple[str | None, tuple[str, ...]]:
        """Find the held skill closest to ``target`` within ``max_hops``.

        Returns ``(bridge_skill, path)``. This is what turns a blunt "missing"
        verdict into the far more useful "transferable - they know Django, so
        Flask is a short hop".
        """
        best: tuple[int, str, tuple[str, ...]] | None = None
        for candidate in held:
            canonical = self.canonical(candidate) or normalise(candidate)
            path = self.shortest_path(canonical, target, max_hops=max_hops)
            if not path:
                continue
            hops = len(path) - 1
            if best is None or hops < best[0] or (hops == best[0] and canonical < best[1]):
                best = (hops, canonical, path)
        if best is None:
            return None, ()
        return best[1], best[2]


_CACHED: SkillOntology | None = None


def get_ontology(path: str | Path | None = None) -> SkillOntology:
    """Return a process-wide cached ontology.

    Parsing the YAML is cheap but happens per resume in a screening loop, so
    caching keeps the hot path free of file I/O.
    """
    global _CACHED
    if path is not None:
        return SkillOntology.load(path)
    if _CACHED is None:
        _CACHED = SkillOntology.load()
    return _CACHED
