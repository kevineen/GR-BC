import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from lerobot.policies.grbc.config import PRESETS, GraphConfig, resolve_min_camera_hits
from lerobot.policies.grbc.export import run as export_run
from lerobot.policies.grbc.graph import build_multi_view_adaptive_similarity_edges

TWO_CAMERAS = ["observation.images.front", "observation.images.wrist"]


def fixture(root, cameras):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "total_frames": 6,
        "total_episodes": 2,
        "fps": 10,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {camera: {"dtype": "video", "info": {"video.fps": 10}} for camera in cameras},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    for ep in range(2):
        data = pd.DataFrame(
            {
                "index": np.arange(3 * ep, 3 * ep + 3),
                "episode_index": ep,
                "frame_index": np.arange(3),
                "timestamp": np.arange(3) / 10,
                "observation.state": [np.array([0.0, 1.0])] * 3,
                "action": [np.array([0.0, 0.0])] * 3,
                "next.reward": [0.0, 1.0, 0.0] if ep == 0 else [0.0, 0.0, 0.0],
            }
        )
        data.to_parquet(root / f"data/chunk-000/file-{ep:03d}.parquet")
        episode = {"episode_index": ep}
        for camera in cameras:
            episode.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": ep * 0.3,
                }
            )
        pd.DataFrame([episode]).to_parquet(root / f"meta/episodes/chunk-000/file-{ep:03d}.parquet")
    return info


class ResolveMinCameraHitsTest(unittest.TestCase):
    def test_none_means_all_cameras(self):
        self.assertEqual(resolve_min_camera_hits(None, 3), 3)
        self.assertEqual(resolve_min_camera_hits(None, 2), 2)
        self.assertEqual(resolve_min_camera_hits(None, 1), 1)

    def test_explicit_value_within_range(self):
        self.assertEqual(resolve_min_camera_hits(2, 3), 2)
        self.assertEqual(resolve_min_camera_hits(1, 1), 1)

    def test_out_of_range_rejected(self):
        for hits, camera_count in ((3, 2), (0, 3), (-1, 2), (4, 3)):
            with self.assertRaises(ValueError):
                resolve_min_camera_hits(hits, camera_count)

    def test_config_accepts_none_and_rejects_nonpositive(self):
        GraphConfig(min_camera_hits=None).validate()
        GraphConfig(min_camera_hits=2).validate()
        with self.assertRaises(ValueError):
            GraphConfig(min_camera_hits=0).validate()

    def test_presets_serialize_none(self):
        for preset in PRESETS.values():
            config = GraphConfig(**{**vars(GraphConfig()), **preset})
            self.assertIsNone(config.min_camera_hits)


class TwoViewGraphTest(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.DataFrame(
            {
                "node_id": np.arange(6),
                "episode_index": [0, 0, 0, 1, 1, 1],
                "frame_index": [0, 10, 20, 0, 10, 20],
            }
        )

    def test_two_camera_intersection_is_subset_of_union(self):
        features = np.random.default_rng(11).normal(size=(6, 8)).astype(np.float32)
        common = {
            "features": features,
            "nodes": self.nodes,
            "gripper_closed": np.zeros(6, dtype=bool),
            "mutual_k": 2,
            "stride": 10,
            "adaptive_alpha": 100,
            "adaptive_base_scale": 0.05,
            "cost_scale": 1,
            "camera_count": 2,
        }
        intersection, stats = build_multi_view_adaptive_similarity_edges(**common, min_camera_hits=2)
        union, _ = build_multi_view_adaptive_similarity_edges(**common, min_camera_hits=1)
        self.assertEqual(stats["camera_count"], 2)
        self.assertTrue({e[:2] for e in intersection}.issubset({e[:2] for e in union}))

    def test_hits_above_camera_count_rejected(self):
        features = np.random.default_rng(11).normal(size=(6, 8)).astype(np.float32)
        with self.assertRaises(ValueError):
            build_multi_view_adaptive_similarity_edges(
                features,
                self.nodes,
                np.zeros(6, dtype=bool),
                mutual_k=2,
                stride=10,
                adaptive_alpha=100,
                adaptive_base_scale=0.05,
                cost_scale=1,
                camera_count=2,
                min_camera_hits=3,
            )


class TwoViewExportTest(unittest.TestCase):
    def test_end_to_end_two_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            root.mkdir()
            fixture(root, TWO_CAMERAS)

            def fake_extract(dataset, nodes, episodes, info, cameras, checkpoint, cache, batch_size=48, device="cuda", reuse=False):
                rng = np.random.default_rng(3)
                features = rng.normal(size=(len(nodes), 2048 * len(cameras))).astype(np.float32)
                return features, {"cache_hit": False, "fingerprint": "test"}

            output = Path(tmp) / "out"
            config = GraphConfig(stride=1, chunk=2, n_action_step=2)
            with patch("lerobot.policies.grbc.export.extract_features", side_effect=fake_extract):
                summary = export_run(root, output, config, "unused.pth", TWO_CAMERAS)

            self.assertEqual(summary["cameras"], TWO_CAMERAS)
            self.assertEqual(summary["edge_filter"]["camera_count"], 2)
            self.assertEqual(summary["edge_filter"]["candidate_min_camera_hits"], 2)
            table = pd.read_parquet(output / "transition_weights.parquet")
            self.assertIn("graph_transition_weight", table)
            self.assertTrue(np.isfinite(table.graph_transition_weight).all())
            self.assertTrue((output / "summary.json").exists())

    def test_duplicate_cameras_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            root.mkdir()
            fixture(root, TWO_CAMERAS)
            with self.assertRaises(ValueError):
                export_run(root, Path(tmp) / "out", GraphConfig(), "unused.pth", TWO_CAMERAS[:1] * 2)


if __name__ == "__main__":
    unittest.main()
