"""Dependency-free identity graph experiment; no biometric inference is performed here."""
import argparse
import copy
import json
import math
from pathlib import Path


class IdentityGraph:
    def __init__(self, data=None):
        self.data = copy.deepcopy(data) if data is not None else {
            "version": 1, "observations": {}, "links": {}, "audit": []}

    def observe(self, observation):
        o = copy.deepcopy(observation)
        required = ("id", "modality", "source", "start", "end", "model", "embedding")
        if any(k not in o for k in required):
            raise ValueError(f"Required fields: {required}")
        if o["modality"] not in ("face", "voice"):
            raise ValueError("modality must be face or voice")
        if any(not isinstance(o[k], str) or not o[k].strip() for k in ("id", "source", "model")):
            raise ValueError("id, source and model must be nonempty strings")
        if not all(isinstance(o[k], (float, int)) and math.isfinite(o[k]) for k in ("start", "end")):
            raise ValueError("timestamps must be finite seconds")
        if not 0 <= o["start"] <= o["end"]:
            raise ValueError("invalid time interval")
        v = o["embedding"]
        if not isinstance(v, list) or not v or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
            raise ValueError("embedding must be a nonempty finite numeric vector")
        norm = math.hypot(*v)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("invalid embedding norm")
        o["embedding"] = [x / norm for x in v]
        if o["id"] in self.data["observations"]:
            if self.data["observations"][o["id"]] != o:
                raise ValueError("observation ID already exists with different data")
            return
        self.data["observations"][o["id"]] = o
        self.data["audit"].append({"action": "observe", "id": o["id"]})

    def candidates(self, observation_id):
        """Rank same-modality/same-model observations. Scores are not probabilities."""
        source = self.data["observations"][observation_id]
        results = []
        for key, target in self.data["observations"].items():
            if key == observation_id or (source["modality"], source["model"]) != (target["modality"], target["model"]):
                continue
            if len(source["embedding"]) != len(target["embedding"]):
                continue
            score = sum(a * b for a, b in zip(source["embedding"], target["embedding"]))
            results.append({"observation_id": key, "cosine": score})
        return sorted(results, key=lambda r: (-r["cosine"], r["observation_id"]))

    def propose(self, link):
        link = copy.deepcopy(link)
        if any(k not in link for k in ("id", "left", "right", "method", "evidence")):
            raise ValueError("link requires id, left, right, method, evidence")
        if not all(isinstance(link[k], str) and link[k].strip() for k in ("id", "left", "right", "method")):
            raise ValueError("link identifiers and method must be nonempty strings")
        if link["id"] in self.data["links"]:
            raise ValueError("link ID already exists")
        if link["left"] == link["right"] or any(link[k] not in self.data["observations"] for k in ("left", "right")):
            raise ValueError("link must reference two different existing observations")
        if not isinstance(link["evidence"], dict) or not link["evidence"]:
            raise ValueError("structured evidence required")
        left, right = (self.data["observations"][link[k]] for k in ("left", "right"))
        if link["method"] == "embedding" and (left["modality"], left["model"], len(left["embedding"])) != (right["modality"], right["model"], len(right["embedding"])):
            raise ValueError("cannot compare embeddings across modalities/models/dimensions")
        link["status"] = "candidate"
        self.data["links"][link["id"]] = link
        self.data["audit"].append({"action": "propose", "id": link["id"]})

    def review(self, link_id, status, reviewer, reason):
        if status not in ("confirmed", "rejected", "revoked") or not reviewer or not reason:
            raise ValueError("explicit review status, reviewer and reason required")
        link = self.data["links"][link_id]
        previous = link["status"]
        link["status"] = status
        self.data["audit"].append({"action": "review", "id": link_id,
            "previous": previous, "status": status, "reviewer": reviewer, "reason": reason})

    def people(self):
        """Rebuild from confirmed edges, retaining each component's oldest anchor.

        IDs of absorbed components become aliases. On a split the old anchor ID
        reappears; consumers must retain graph revision and observation IDs.
        """
        observations = self.data["observations"]
        parent = {key: key for key in observations}
        order = {key: i for i, key in enumerate(observations)}

        def root(key):
            while parent[key] != key:
                key = parent[key]
            return key

        for link in self.data["links"].values():
            if link["status"] != "confirmed":
                continue
            a, b = root(link["left"]), root(link["right"])
            if order[a] > order[b]:
                a, b = b, a
            parent[b] = a
        people = {}
        for key, observation in observations.items():
            pid = "person:" + root(key)
            person = people.setdefault(pid, {"person_id": pid, "aliases": [], "faces": [], "voices": []})
            if key != root(key):
                person["aliases"].append("person:" + key)
            person[observation["modality"] + "s"].append(key)
        return list(people.values())

    def export(self):
        people = self.people()
        mapping = {key: p["person_id"] for p in people for key in p["faces"] + p["voices"]}
        return {"schema_version": 1, "revision": len(self.data["audit"]), "people": people,
            "events": [{"person_id": mapping[o["id"]], "observation_id": o["id"],
                "modality": o["modality"], "source": o["source"], "start": o["start"],
                "end": o["end"], "text": o.get("text", "")}
                for o in self.data["observations"].values()],
            "links": list(self.data["links"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON fixture with observations and optional links/reviews")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    fixture = json.loads(args.input.read_text())
    graph = IdentityGraph()
    for observation in fixture["observations"]:
        graph.observe(observation)
    for link in fixture.get("links", []):
        graph.propose(link)
    for review in fixture.get("reviews", []):
        graph.review(**review)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, data in (("graph.json", graph.data), ("people.json", graph.export())):
        (args.out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"{len(graph.people())} people; saved to {args.out}")


if __name__ == "__main__":
    main()
