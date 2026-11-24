import json
import unittest
from identity import IdentityGraph


def observation(key, modality="face", vector=None, model="test-v1"):
    return {"id": key, "modality": modality, "source": "clip-a.mp4",
            "start": 0, "end": 2, "model": model, "embedding": vector or [1, 0]}


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.g = IdentityGraph()
        for key, modality in (("f1", "face"), ("v1", "voice"), ("f2", "face")):
            self.g.observe(observation(key, modality))

    def propose(self, key, left, right, method="manual"):
        self.g.propose({"id": key, "left": left, "right": right, "method": method,
                        "evidence": {"source": "clip-a.mp4", "start": 0, "end": 2}})

    def test_cross_modality_embeddings_never_compared(self):
        self.assertEqual([r["observation_id"] for r in self.g.candidates("f1")], ["f2"])
        with self.assertRaises(ValueError):
            self.propose("bad", "f1", "v1", "embedding")

    def test_review_and_revoke_restore_identity(self):
        self.propose("link", "f1", "v1")
        self.assertEqual(len(self.g.people()), 3)
        self.g.review("link", "confirmed", "tester", "verified video")
        self.assertEqual(len(self.g.people()), 2)
        self.assertEqual(self.g.people()[0]["person_id"], "person:f1")
        self.g.review("link", "revoked", "tester", "wrong speaker")
        self.assertEqual(len(self.g.people()), 3)

    def test_transitive_link_revocation(self):
        self.propose("a", "f1", "v1")
        self.propose("b", "v1", "f2")
        for key in ("a", "b"):
            self.g.review(key, "confirmed", "tester", "verified")
        self.assertEqual(len(self.g.people()), 1)
        self.g.review("b", "revoked", "tester", "correction")
        self.assertEqual(len(self.g.people()), 2)
        self.assertEqual(self.g.people()[0]["voices"], ["v1"])

    def test_reload_and_new_observation_preserve_ids(self):
        restored = IdentityGraph(json.loads(json.dumps(self.g.data)))
        restored.observe(observation("f3"))
        self.assertEqual(restored.people()[:3], self.g.people())

    def test_invalid_inputs(self):
        for vector in ([0, 0], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                self.g.observe(observation("bad", vector=vector))
        with self.assertRaises(ValueError):
            self.g.observe(observation("f1", vector=[0, 1]))

    def test_model_isolation_and_export_provenance(self):
        self.g.observe(observation("f3", model="different-v1"))
        self.assertNotIn("f3", [r["observation_id"] for r in self.g.candidates("f1")])
        event = self.g.export()["events"][0]
        self.assertEqual((event["source"], event["start"], event["end"]), ("clip-a.mp4", 0, 2))


if __name__ == "__main__":
    unittest.main()
