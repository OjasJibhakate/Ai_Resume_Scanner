"""Skill ontology: alias resolution, one-directional expansion, graph traversal."""

import textwrap

import pytest

from talentlens.ontology import OntologyError, SkillOntology, get_ontology, normalise


def build(yaml_text: str, tmp_path) -> SkillOntology:
    path = tmp_path / "ontology.yaml"
    path.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return SkillOntology.load(path)


class TestAliases:
    def test_canonical_name_resolves_to_itself(self, ontology):
        assert ontology.canonical("python") == "python"

    def test_aliases_resolve(self, ontology):
        assert ontology.canonical("js") == "javascript"
        assert ontology.canonical("ES6") == "javascript"
        assert ontology.canonical("Django REST Framework") == "django"

    def test_resolution_is_case_and_space_insensitive(self, ontology):
        assert ontology.canonical("  MACHINE   LEARNING ") == "machine-learning"

    def test_hyphenated_names_match_their_spaced_spelling(self, ontology):
        """Resumes write 'GitHub Actions', the ontology stores 'github-actions'."""
        assert ontology.canonical("GitHub Actions") == "github-actions"
        assert ontology.canonical("machine learning") == "machine-learning"
        assert ontology.canonical("REST API") == "rest-api"

    def test_unknown_surface_returns_none(self, ontology):
        assert ontology.canonical("underwater basket weaving") is None


class TestExpansion:
    def test_child_implies_parent(self, ontology):
        assert "python" in ontology.expand(["django"])

    def test_expansion_is_transitive(self, ontology):
        expanded = ontology.expand(["keras"])
        assert {"keras", "tensorflow", "deep-learning", "machine-learning"} <= expanded

    def test_parent_does_not_imply_child(self, ontology):
        """The asymmetry that stops a generalist matching every specialist req."""
        assert "django" not in ontology.expand(["python"])

    def test_unknown_skills_are_preserved_not_dropped(self, ontology):
        assert "some-niche-tool" in ontology.expand(["some-niche-tool"])

    def test_empty_input(self, ontology):
        assert ontology.expand([]) == set()


class TestTraversal:
    def test_siblings_share_a_parent(self, ontology):
        assert "flask" in ontology.siblings("django")

    def test_distance_between_siblings_is_two_hops(self, ontology):
        assert ontology.distance("django", "flask") == 2

    def test_distance_returns_none_when_too_far(self, ontology):
        assert ontology.distance("django", "bgp", max_hops=2) is None

    def test_shortest_path_runs_through_the_shared_parent(self, ontology):
        assert ontology.shortest_path("django", "flask") == ("django", "python", "flask")

    def test_nearest_held_finds_the_bridge(self, ontology):
        bridge, path = ontology.nearest_held("flask", ["django", "bgp"])
        assert bridge == "django"
        assert path == ("django", "python", "flask")

    def test_nearest_held_returns_nothing_when_unrelated(self, ontology):
        bridge, path = ontology.nearest_held("bgp", ["react"], max_hops=2)
        assert bridge is None
        assert path == ()


class TestLoading:
    def test_cycles_are_rejected(self, tmp_path):
        with pytest.raises(OntologyError, match="cycle"):
            build(
                """
                skills:
                  a: {parents: [b]}
                  b: {parents: [a]}
                """,
                tmp_path,
            )

    def test_unknown_parent_is_rejected(self, tmp_path):
        with pytest.raises(OntologyError, match="unknown parent"):
            build(
                """
                skills:
                  a: {parents: [ghost]}
                """,
                tmp_path,
            )

    def test_duplicate_alias_across_skills_is_rejected(self, tmp_path):
        with pytest.raises(OntologyError, match="claimed by both"):
            build(
                """
                skills:
                  alpha: {aliases: [shared]}
                  beta: {aliases: [shared]}
                """,
                tmp_path,
            )

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(OntologyError, match="not found"):
            SkillOntology.load(tmp_path / "nope.yaml")

    def test_shipped_ontology_loads_and_is_substantial(self, ontology):
        assert len(ontology) > 100

    def test_get_ontology_is_cached(self):
        assert get_ontology() is get_ontology()


def test_normalise_collapses_whitespace_and_case():
    assert normalise("  Machine   LEARNING  ") == "machine learning"
