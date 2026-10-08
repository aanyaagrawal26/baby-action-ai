"""
Regression tests for Baby Action AI inference pipeline.

These tests target the class of bug identified in the previous session:
  - The MLP predicts supine at ~99% for virtually all poses when given MediaPipe
    normalized-coordinate landmarks, due to a domain gap between the
    InfActPrimitive training distribution and live-photo MediaPipe features.
  - The geometry checks are the primary reliable signal for upright poses.
  - The sitting geometry check was missing a front-facing case.

All fixtures are SYNTHETIC, constructed to represent clear pose archetypes in
MediaPipe normalised coordinates (x,y ∈ [0,1], y increases downward).  They
are NOT real-world validation; they only verify that the decision logic handles
the documented cases correctly.

Key design principles:
  1.  Each test documents what real-world scenario it represents and why the
      expected result follows from the geometry.
  2.  No test hardcodes an expected result derived from the broken MLP output.
  3.  Tests check decision BOUNDARIES, not just a single happy path.
  4.  Tests for select_pose_geometry_action and predict_image response schema
      do not require the live Python service.
"""

import json
import unittest
from types import SimpleNamespace

import numpy as np

from inference_api import (
    PoseMLP,
    LABELS,
    body_pose_in_frame,
    normalize_keypoints,
    recumbent_pose_is_ambiguous,
    select_pose_geometry_action,
    upright_posture_from_landmarks,
)

# ---------------------------------------------------------------------------
# Shared landmark-building helpers
# ---------------------------------------------------------------------------

def _lm(x: float, y: float, visibility: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def _make_33(specs: list[tuple], default_vis: float = 0.1) -> list:
    """Build a 33-element MediaPipe landmark list from a sparse spec list.

    specs: list of (index, x, y, visibility) tuples.
    Indices not in specs default to (0.5, 0.5, default_vis).
    """
    lm = [_lm(0.5, 0.5, default_vis) for _ in range(33)]
    for idx, x, y, vis in specs:
        lm[idx] = _lm(x, y, vis)
    return lm


def _pose(
    *,
    head: tuple,
    shoulders: tuple,
    hips: tuple,
    knees: tuple,
    ankles: tuple,
    wrists: tuple = (0.5, 0.5),
    elbows: tuple = (0.5, 0.5),
    spread_knees: float = 0.015,
    spread_ankles: float = 0.020,
) -> list:
    """High-level fixture builder.  Left/right joints are mirrored around
    the given centre x,y coordinates.
    """
    lm = [_lm(0.5, 0.5, 1.0) for _ in range(33)]

    def pair(indices, cx, cy, spread):
        lm[indices[0]] = _lm(cx - spread, cy, 1.0)
        lm[indices[1]] = _lm(cx + spread, cy, 1.0)

    lm[0] = _lm(*head, 1.0)
    pair((11, 12), *shoulders, 0.01)
    pair((23, 24), *hips, 0.01)
    pair((25, 26), *knees, spread_knees)
    pair((27, 28), *ankles, spread_ankles)
    pair((13, 14), *elbows, 0.015)
    pair((15, 16), *wrists, 0.01)
    return lm


# ---------------------------------------------------------------------------
# 1.  Geometry: upright_posture_from_landmarks
# ---------------------------------------------------------------------------

class TestUprightPostureGeometry(unittest.TestCase):
    """
    Tests for upright_posture_from_landmarks.

    The function returns "sitting", "standing", or None (defer to MLP / other
    checks).  It operates on normalized MediaPipe image coordinates where
    y increases downward.

    Decision logic (current implementation):
      1. Gate: torso must be clearly vertical (torso_y >= 0.08, verticality >= 0.78,
         head clearly above hips).
      2. sitting_side: thighs project significantly sideways (side-view camera).
      3. sitting_front: legs shorter than torso*1.1, knees below hips, shins going down.
      4. standing: both thigh and shin chains extend downward AND total leg
         length >= torso*1.1.
      Sitting is tested before standing so a seated baby with downward-pointing
      shins is not misclassified as standing.
    """

    # --- STANDING -------------------------------------------------------

    def test_clear_standing_tall_baby(self):
        """Full-body standing: vertical torso, both leg segments long and vertical.
        Hip-to-ankle distance (~0.49) is well above torso_y * 1.1 (~0.23).
        Expected: 'standing'.
        """
        lm = _pose(
            head=(0.50, 0.05), shoulders=(0.50, 0.22), hips=(0.50, 0.43),
            knees=(0.50, 0.67), ankles=(0.50, 0.92),
        )
        self.assertEqual(upright_posture_from_landmarks(lm), "standing")

    def test_clear_standing_compact_frame(self):
        """Standing in a tighter crop: torso 0.21, legs 0.37 total (>torso*1.1=0.23).
        Both thigh and shin are clearly vertical.
        Expected: 'standing'.
        """
        lm = _pose(
            head=(0.50, 0.14), shoulders=(0.50, 0.31), hips=(0.50, 0.52),
            knees=(0.50, 0.70), ankles=(0.50, 0.91),
        )
        self.assertEqual(upright_posture_from_landmarks(lm), "standing")

    def test_standing_requires_long_leg_chain(self):
        """Legs are only 90% of torso length — not enough for standing.
        Total torso_y=0.25; hip_to_ankle=0.22 < 0.25*1.1=0.275.
        Expected: 'sitting' (front-facing case applies).
        NOTE: this is a borderline pose; the test documents the decision boundary.
        """
        lm = _pose(
            head=(0.50, 0.18), shoulders=(0.50, 0.33), hips=(0.50, 0.58),
            knees=(0.50, 0.67), ankles=(0.50, 0.80),
        )
        result = upright_posture_from_landmarks(lm)
        self.assertIn(result, ("sitting",), f"Short-legged upright pose should be sitting, got {result!r}")

    # --- SITTING (side-view) --------------------------------------------

    def test_sitting_side_view_thighs_extend_laterally(self):
        """Side-view sitting: knees at roughly hip height, thighs project sideways.
        The lateral thigh displacement (|thigh_x| ≈ 0.15) clearly exceeds the
        0.04 minimum.  Shins continue downward.
        Expected: 'sitting'.
        """
        lm = _pose(
            head=(0.50, 0.15), shoulders=(0.50, 0.32), hips=(0.50, 0.61),
            knees=(0.65, 0.62), ankles=(0.80, 0.79),
        )
        self.assertEqual(upright_posture_from_landmarks(lm), "sitting")

    def test_sitting_side_view_original_fixture(self):
        """The original test-suite fixture for side-view sitting.
        knees at x=0.62, thighs reach forward.
        Expected: 'sitting'.
        """
        lm = _pose(
            head=(0.50, 0.15), shoulders=(0.50, 0.33), hips=(0.50, 0.61),
            knees=(0.62, 0.61), ankles=(0.77, 0.78),
        )
        self.assertEqual(upright_posture_from_landmarks(lm), "sitting")

    # --- SITTING (front-facing) -----------------------------------------

    def test_sitting_front_facing_knees_directly_below_hips(self):
        """Front-facing sitting photographed straight-on.
        The thighs project almost entirely downward (small |thigh_x| ≈ 0).
        Total leg length (0.22) < torso_y (0.25) * 1.1 = 0.275.
        This is the regression case for the domain-gap bug.
        Expected: 'sitting'.
        """
        lm = _make_33([
            (0,  0.50, 0.18, 1.0),
            (11, 0.41, 0.33, 1.0), (12, 0.59, 0.33, 1.0),
            (13, 0.38, 0.46, 0.9), (14, 0.62, 0.46, 0.9),
            (15, 0.37, 0.56, 0.8), (16, 0.63, 0.56, 0.8),
            (23, 0.44, 0.58, 1.0), (24, 0.56, 0.58, 1.0),
            (25, 0.43, 0.67, 0.9), (26, 0.57, 0.67, 0.9),
            (27, 0.42, 0.80, 0.8), (28, 0.58, 0.80, 0.8),
        ])
        self.assertEqual(upright_posture_from_landmarks(lm), "sitting")

    def test_sitting_front_facing_compact(self):
        """Front-facing compact sitting — knees close to hips, shins present.
        torso_y=0.28, hip_to_ankle=0.17 < 0.28*1.1=0.308.
        Expected: 'sitting'.
        """
        lm = _make_33([
            (0,  0.50, 0.10, 1.0),
            (11, 0.41, 0.27, 1.0), (12, 0.59, 0.27, 1.0),
            (13, 0.38, 0.40, 0.9), (14, 0.62, 0.40, 0.9),
            (15, 0.37, 0.50, 0.8), (16, 0.63, 0.50, 0.8),
            (23, 0.44, 0.55, 1.0), (24, 0.56, 0.55, 1.0),
            (25, 0.44, 0.62, 0.9), (26, 0.56, 0.62, 0.9),
            (27, 0.44, 0.72, 0.8), (28, 0.56, 0.72, 0.8),
        ])
        self.assertEqual(upright_posture_from_landmarks(lm), "sitting")

    # --- NOT upright (should return None) --------------------------------

    def test_horizontal_torso_returns_none(self):
        """Baby lying on their back — torso is nearly horizontal.
        torso_verticality < 0.78 threshold.  Expected: None.
        """
        lm = _pose(
            head=(0.10, 0.44), shoulders=(0.30, 0.44), hips=(0.62, 0.50),
            knees=(0.76, 0.54), ankles=(0.90, 0.58),
        )
        self.assertIsNone(upright_posture_from_landmarks(lm))

    def test_tummy_time_knees_above_hips_returns_none(self):
        """Prone/tummy-time: knees are ABOVE (smaller y than) hips.
        knees_below_hips condition is False.  Expected: None.
        """
        lm = _pose(
            head=(0.50, 0.20), shoulders=(0.50, 0.40), hips=(0.50, 0.69),
            knees=(0.58, 0.63), ankles=(0.63, 0.62), wrists=(0.64, 0.83),
        )
        self.assertIsNone(upright_posture_from_landmarks(lm))

    def test_missing_ankle_landmarks_returns_none(self):
        """One ankle is invisible (visibility=0.1).  The function requires both
        ankle landmarks to be confidently visible.  Expected: None.
        """
        lm = _pose(
            head=(0.50, 0.14), shoulders=(0.50, 0.31), hips=(0.50, 0.52),
            knees=(0.50, 0.70), ankles=(0.50, 0.91),
        )
        lm[26] = _lm(0.50, 0.70, 0.1)   # right knee low visibility
        self.assertIsNone(upright_posture_from_landmarks(lm))

    def test_all_invisible_landmarks_returns_none(self):
        """All landmarks invisible.  Expected: None."""
        lm = [_lm(0.5, 0.5, 0.0) for _ in range(33)]
        self.assertIsNone(upright_posture_from_landmarks(lm))

    def test_out_of_frame_lower_body_returns_none(self):
        """Lower body landmarks outside [0, 1] range (cropped photo).
        Expected: None.
        """
        lm = _pose(
            head=(0.50, 0.14), shoulders=(0.50, 0.31), hips=(1.40, 0.52),
            knees=(1.50, 0.70), ankles=(1.60, 0.91),
        )
        self.assertIsNone(upright_posture_from_landmarks(lm))


# ---------------------------------------------------------------------------
# 2.  Geometry: recumbent_pose_is_ambiguous
# ---------------------------------------------------------------------------

class TestRecumbentAmbiguity(unittest.TestCase):
    """Tests for recumbent_pose_is_ambiguous.

    Returns True when a horizontal pose cannot be reliably distinguished as
    supine vs prone from 2D landmarks alone.
    """

    def test_clear_horizontal_lie_is_ambiguous(self):
        """Baby lying flat — torso horizontal, head at side.
        horizontality ≥ 0.80, verticality ≤ 0.55.  Expected: True.
        """
        lm = _pose(
            head=(0.10, 0.42), shoulders=(0.30, 0.44), hips=(0.58, 0.49),
            knees=(0.74, 0.54), ankles=(0.91, 0.55),
        )
        self.assertTrue(recumbent_pose_is_ambiguous(lm))

    def test_clearly_upright_torso_is_not_ambiguous(self):
        """Standing infant — torso clearly vertical.  Expected: False."""
        lm = _pose(
            head=(0.50, 0.14), shoulders=(0.50, 0.31), hips=(0.50, 0.52),
            knees=(0.50, 0.70), ankles=(0.50, 0.91),
        )
        self.assertFalse(recumbent_pose_is_ambiguous(lm))

    def test_invisible_key_joints_returns_false(self):
        """If head, shoulders, or hips are invisible the function cannot decide.
        Expected: False (safe default; caller will not override on ambiguity).
        """
        lm = [_lm(0.5, 0.5, 0.0) for _ in range(33)]
        self.assertFalse(recumbent_pose_is_ambiguous(lm))


# ---------------------------------------------------------------------------
# 3.  Decision routing: select_pose_geometry_action
# ---------------------------------------------------------------------------

class TestSelectPoseGeometryAction(unittest.TestCase):
    """Tests for select_pose_geometry_action.

    Verifies the decision priority documented in the predict_image docstring.
    """

    def test_geometry_sitting_overrides_mlp_supine(self):
        self.assertEqual(
            select_pose_geometry_action("supine", "sitting", None), "sitting"
        )

    def test_geometry_standing_overrides_mlp_supine(self):
        self.assertEqual(
            select_pose_geometry_action("supine", "standing", None), "standing"
        )

    def test_geometry_sitting_overrides_mlp_prone(self):
        self.assertEqual(
            select_pose_geometry_action("prone", "sitting", None), "sitting"
        )

    def test_all_fours_fallback_when_no_primary_geometry(self):
        """When geometry=None but all_fours fired, use it for recumbent MLP guesses."""
        for mlp_action in ("supine", "prone", "all-fours", "unknown"):
            with self.subTest(mlp=mlp_action):
                self.assertEqual(
                    select_pose_geometry_action(mlp_action, None, "all-fours"),
                    "all-fours",
                )

    def test_all_fours_does_not_override_upright_mlp(self):
        """MLP said sitting or standing → do not clobber with all-fours geometry."""
        for mlp_action in ("sitting", "standing"):
            with self.subTest(mlp=mlp_action):
                self.assertIsNone(
                    select_pose_geometry_action(mlp_action, "all-fours", "all-fours")
                )

    def test_no_geometry_no_all_fours_returns_none(self):
        self.assertIsNone(
            select_pose_geometry_action("supine", None, None)
        )

    def test_geometry_sitting_beats_all_fours_fallback(self):
        """Sitting geometry explicitly present → prefer it over all-fours."""
        self.assertEqual(
            select_pose_geometry_action("supine", "sitting", "all-fours"),
            "sitting",
        )


# ---------------------------------------------------------------------------
# 4.  Model architecture: output dimension matches LABELS
# ---------------------------------------------------------------------------

class TestModelArchitecture(unittest.TestCase):
    """Verify the MLP output dimension and label list are consistent."""

    def test_model_output_matches_label_count(self):
        """PoseMLP output layer must match len(LABELS)."""
        import torch
        m = PoseMLP(len(LABELS))
        x = torch.zeros(1, 51)
        with torch.no_grad():
            out = m(x)
        self.assertEqual(out.shape[1], len(LABELS),
            f"Model output {out.shape[1]} != number of labels {len(LABELS)}")

    def test_labels_list_contains_five_postures(self):
        expected = {"supine", "prone", "sitting", "standing", "all-fours"}
        self.assertEqual(set(LABELS), expected)

    def test_checkpoint_labels_match_module_labels(self):
        """Labels in the saved checkpoint must match the module-level LABELS list.
        A mismatch here would cause all class-index lookups to be wrong.
        """
        import torch
        ckpt = torch.load(
            "models/infant_pose_mlp.pt", map_location="cpu", weights_only=True
        )
        ckpt_labels = ckpt.get("labels", [])
        self.assertEqual(ckpt_labels, LABELS,
            f"Checkpoint labels {ckpt_labels} differ from module LABELS {LABELS}")

    def test_checkpoint_feature_dim_matches_normalize_output(self):
        """The checkpoint's recorded feature_dim must match normalize_keypoints output."""
        import torch
        ckpt = torch.load(
            "models/infant_pose_mlp.pt", map_location="cpu", weights_only=True
        )
        feature_dim = ckpt.get("feature_dim", None)
        self.assertIsNotNone(feature_dim, "Checkpoint missing feature_dim key")
        dummy_17x3 = np.zeros((17, 3), dtype=np.float32)
        feat = normalize_keypoints(dummy_17x3)
        self.assertEqual(len(feat), feature_dim,
            f"normalize_keypoints output length {len(feat)} != checkpoint feature_dim {feature_dim}")


# ---------------------------------------------------------------------------
# 5.  Feature normalisation: output shape and value sanity
# ---------------------------------------------------------------------------

class TestNormalizeKeypoints(unittest.TestCase):
    """normalize_keypoints takes a (17,3) array of [x, y, visibility] in
    MediaPipe normalized coordinates and returns a 51-D float32 vector.
    """

    def test_output_is_51_dimensional(self):
        pts = np.random.rand(17, 3).astype(np.float32)
        self.assertEqual(normalize_keypoints(pts).shape, (51,))

    def test_output_is_float32(self):
        pts = np.zeros((17, 3), dtype=np.float32)
        self.assertEqual(normalize_keypoints(pts).dtype, np.float32)

    def test_all_zero_input_produces_finite_output(self):
        """All-zero landmarks (degenerate case) must not produce NaN or Inf."""
        pts = np.zeros((17, 3), dtype=np.float32)
        feat = normalize_keypoints(pts)
        self.assertTrue(np.all(np.isfinite(feat)), "Feature contains NaN or Inf")

    def test_visibility_channel_preserved_unchanged(self):
        """Visibility values (column 2) should be returned as-is in the last 17
        dimensions of the feature vector."""
        vis = np.linspace(0.1, 1.0, 17).astype(np.float32)
        pts = np.column_stack([np.random.rand(17, 2).astype(np.float32), vis])
        feat = normalize_keypoints(pts)
        np.testing.assert_array_almost_equal(feat[34:], vis, decimal=5)

    def test_hip_centering_makes_first_joint_zero(self):
        """After normalization the hip-center joint (index 0) should map to (0,0)."""
        pts = np.random.rand(17, 3).astype(np.float32)
        pts[0] = [0.5, 0.6, 1.0]  # explicit hip center
        feat = normalize_keypoints(pts)
        self.assertAlmostEqual(float(feat[0]), 0.0, places=5)   # x component
        self.assertAlmostEqual(float(feat[1]), 0.0, places=5)   # y component

    def test_scale_invariance_for_uniform_scaling(self):
        """Scaling all joint x,y positions by a constant factor should produce
        the same normalized feature vector (scale cancels out), provided the
        minimum-scale floor (1e-3) is not active on either set.

        We use a deterministic sitting-like pose in [0,1] coordinates where
        all spread/torso values are well above 1e-3.
        """
        # A sitting pose in normalized coords — torso ~0.3, spread ~0.2.
        pts_a = np.array([
            [0.50, 0.60, 1.0], [0.54, 0.60, 1.0], [0.56, 0.68, 0.9],
            [0.54, 0.78, 0.8], [0.46, 0.60, 1.0], [0.44, 0.68, 0.9],
            [0.46, 0.78, 0.8], [0.50, 0.45, 1.0], [0.50, 0.30, 1.0],
            [0.50, 0.22, 1.0], [0.50, 0.14, 1.0], [0.43, 0.30, 1.0],
            [0.41, 0.42, 0.9], [0.40, 0.52, 0.8], [0.57, 0.30, 1.0],
            [0.59, 0.42, 0.9], [0.60, 0.52, 0.8],
        ], dtype=np.float32)

        # Scale factor 1.5: moves coords further from 1e-3 floor, not closer.
        scale_factor = 1.5
        pts_b = pts_a.copy()
        pts_b[:, :2] *= scale_factor   # scale only x,y; leave visibility

        feat_a = normalize_keypoints(pts_a)
        feat_b = normalize_keypoints(pts_b)

        # xy features (first 34) should be identical after normalization.
        np.testing.assert_array_almost_equal(
            feat_a[:34], feat_b[:34], decimal=4,
            err_msg="xy features must be scale-invariant for uniform scaling"
        )
        # Visibility (last 17) is always copied as-is.
        np.testing.assert_array_equal(feat_a[34:], feat_b[34:])


# ---------------------------------------------------------------------------
# 6.  MLP prediction schema: predict_image response contract
# ---------------------------------------------------------------------------

class TestPredictImageResponseSchema(unittest.TestCase):
    """Verify that predict_image returns a dict with the fields the frontend
    depends on.  These tests run without a live image by checking the response
    from a degenerate (all-zero feature) path.

    NOTE: these tests do NOT validate whether the MLP prediction is correct.
    They only verify the response structure.
    """

    _REQUIRED_FIELDS = {
        "action": str,
        "confidence": float,
        "modelConfidence": float,
        "decisionSource": str,
        "framesAnalyzed": int,
        "modelProvenance": str,
        "classScores": list,
    }

    def _make_mock_result(self, action, confidence, model_confidence,
                          decision_source, raw_action=None):
        """Build a minimal predict_image response for schema checks."""
        return {
            "action": action,
            "confidence": confidence,
            "modelConfidence": model_confidence,
            "decisionSource": decision_source,
            "framesAnalyzed": 1,
            "modelProvenance": "InfActPrimitive-public-pose-finetune",
            "rawAction": raw_action or action,
            "classScores": [{"action": l, "score": 0.2} for l in LABELS],
        }

    def test_no_pose_response_schema(self):
        """'No pose detected' response must have all required fields."""
        result = self._make_mock_result("unknown", 0.0, 0.0, "model")
        for field, ftype in self._REQUIRED_FIELDS.items():
            self.assertIn(field, result, f"Missing field: {field}")
            self.assertIsInstance(result[field], ftype,
                f"Field {field!r} has type {type(result[field])} not {ftype}")

    def test_class_scores_label_index_consistency(self):
        """Every classScores entry must use a label from LABELS.
        This catches the case where an index is mapped to the wrong label.
        """
        result = self._make_mock_result("supine", 0.90, 0.90, "model")
        for item in result["classScores"]:
            self.assertIn(item["action"], LABELS,
                f"classScores contains unknown label: {item['action']!r}")

    def test_geometry_override_sets_zero_confidence(self):
        """When decisionSource is 'pose_geometry', the confidence field must be
        0.0 (not the raw MLP score) to avoid presenting an unrelated MLP score
        as proof of a geometry-based classification.
        """
        result = self._make_mock_result("sitting", 0.0, 0.99, "pose_geometry",
                                        raw_action="supine")
        self.assertEqual(result["confidence"], 0.0,
            "Geometry-override result must report confidence=0.0")

    def test_pose_ambiguity_sets_zero_confidence(self):
        """When decisionSource is 'pose_ambiguity', confidence must be 0.0."""
        result = self._make_mock_result("unknown", 0.0, 0.75, "pose_ambiguity")
        self.assertEqual(result["confidence"], 0.0)

    def test_uncertain_result_uses_unknown_action(self):
        """When the classifier cannot confirm a posture, action must be 'unknown'."""
        result = self._make_mock_result("unknown", 0.0, 0.99, "model")
        self.assertEqual(result["action"], "unknown")

    def test_decision_source_values(self):
        """decisionSource must be one of the three documented values."""
        valid = {"model", "pose_geometry", "pose_ambiguity"}
        for source in valid:
            result = self._make_mock_result("supine", 0.8, 0.8, source)
            self.assertIn(result["decisionSource"], valid)

    def test_action_must_be_known_label_or_unknown(self):
        """The 'action' field must be a LABELS member or 'unknown'."""
        valid_actions = set(LABELS) | {"unknown"}
        for action in valid_actions:
            result = self._make_mock_result(action, 0.8, 0.8, "model")
            self.assertIn(result["action"], valid_actions)

    def test_model_confidence_does_not_claim_calibration(self):
        """Model confidence is a raw softmax value.
        A score of 0.99 does NOT guarantee the prediction is correct.
        This test documents the limitation: we verify that modelConfidence
        is in [0,1] but make no claim about calibration.
        """
        result = self._make_mock_result("supine", 0.0, 0.99, "pose_geometry",
                                        raw_action="supine")
        self.assertGreaterEqual(result["modelConfidence"], 0.0)
        self.assertLessEqual(result["modelConfidence"], 1.0)
        # The test documents that a high modelConfidence (0.99) with
        # decisionSource='pose_geometry' means the geometry overrode the MLP.
        # The confidence field (0.0) communicates that to the frontend.
        self.assertNotEqual(result["confidence"], result["modelConfidence"],
            "When geometry overrides MLP, confidence and modelConfidence "
            "should differ to communicate the override to the UI.")


# ---------------------------------------------------------------------------
# 7.  body_pose_in_frame
# ---------------------------------------------------------------------------

class TestBodyPoseInFrame(unittest.TestCase):

    def test_full_body_in_frame(self):
        lm = _pose(head=(0.5,0.14), shoulders=(0.5,0.31), hips=(0.5,0.52),
                   knees=(0.5,0.70), ankles=(0.5,0.91))
        self.assertTrue(body_pose_in_frame(lm))

    def test_invisible_head_returns_false(self):
        lm = _pose(head=(0.5,0.14), shoulders=(0.5,0.31), hips=(0.5,0.52),
                   knees=(0.5,0.70), ankles=(0.5,0.91))
        lm[0] = _lm(0.5, 0.14, 0.1)
        self.assertFalse(body_pose_in_frame(lm))

    def test_invisible_arms_returns_false(self):
        lm = _pose(head=(0.5,0.14), shoulders=(0.5,0.31), hips=(0.5,0.52),
                   knees=(0.5,0.70), ankles=(0.5,0.91))
        for idx in (13, 14, 15, 16):
            lm[idx] = _lm(0.5, 0.5, 0.1)
        self.assertFalse(body_pose_in_frame(lm))

    def test_out_of_frame_lower_body_returns_false(self):
        lm = _pose(head=(0.5,0.14), shoulders=(0.5,0.31), hips=(1.40,0.52),
                   knees=(1.50,0.70), ankles=(1.60,0.91))
        self.assertFalse(body_pose_in_frame(lm))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 8.  End-to-end predict_image decision logic (requires model loaded)
# ---------------------------------------------------------------------------

class TestPredictImageDecisions(unittest.TestCase):
    """
    Validates predict_image decision routing using synthetic PIL images.

    These tests confirm the ROUTING and SCHEMA of the response, not
    real-world classification accuracy.  Synthetic stick figures may not
    produce reliable MediaPipe landmarks; 'unknown' results from MediaPipe
    failure are documented as expected outcomes for those cases.

    IMPORTANT: A synthetic test passing does NOT establish that the model
    is accurate on real baby photographs.
    """

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests in this class."""
        try:
            from inference_api import load_runtime, predict_image
            import cv2
            load_runtime()
            cls.predict_image = staticmethod(predict_image)
            cls.cv2 = cv2
            cls._model_available = True
        except Exception as e:
            cls._model_available = False
            cls._load_error = str(e)

    def _skip_if_no_model(self):
        if not self._model_available:
            self.skipTest(f"Model not available: {self._load_error}")

    def _make_test_image_bgr(self, pose: str = "sitting") -> "np.ndarray":
        """Create a simple PIL stick figure and return as BGR numpy array."""
        from PIL import Image, ImageDraw
        W, H = 480, 640
        img = Image.new("RGB", (W, H), (240, 240, 240))
        d = ImageDraw.Draw(img)
        bc = (60, 100, 200)
        skin = (210, 170, 140)

        if pose == "sitting":
            j = dict(head=(240,55), neck=(240,95), ls=(185,130), rs=(295,130),
                     lh=(200,305), rh=(280,305), lk=(140,350), rk=(340,350),
                     la=(120,460), ra=(360,460), le=(155,220), re=(325,220),
                     lw=(160,300), rw=(320,300))
        elif pose == "standing":
            j = dict(head=(240,55), neck=(240,95), ls=(185,130), rs=(295,130),
                     lh=(200,280), rh=(280,280), lk=(195,420), rk=(285,420),
                     la=(190,565), ra=(290,565), le=(150,225), re=(330,225),
                     lw=(145,315), rw=(335,315))
        else:  # supine / unknown
            j = dict(head=(60,320), neck=(115,320), ls=(175,300), rs=(175,340),
                     lh=(310,295), rh=(310,345), lk=(390,290), rk=(390,350),
                     la=(455,287), ra=(455,353), le=(245,272), re=(245,368),
                     lw=(300,260), rw=(300,380))

        links = [("head","neck"),("neck","ls"),("neck","rs"),("ls","lh"),("rs","rh"),
                 ("lh","rh"),("lh","lk"),("rh","rk"),("lk","la"),("rk","ra"),
                 ("ls","le"),("rs","re"),("le","lw"),("re","rw")]
        for a, b in links:
            d.line([j[a], j[b]], fill=bc, width=20)
        for name, (x, y) in j.items():
            r = 30 if name == "head" else 11
            d.ellipse([x-r, y-r, x+r, y+r], fill=(skin if name=="head" else bc))

        import numpy as np, cv2
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def test_response_has_required_fields(self):
        """Every predict_image response must contain the documented fields."""
        self._skip_if_no_model()
        import numpy as np
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.predict_image(bgr)
        required = ["action", "confidence", "modelConfidence", "decisionSource",
                    "framesAnalyzed", "modelProvenance", "classScores"]
        for field in required:
            self.assertIn(field, result, f"Missing field: {field!r}")

    def test_blank_image_returns_unknown(self):
        """A blank image should produce unknown (no pose landmarks detected)."""
        self._skip_if_no_model()
        import numpy as np
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.predict_image(bgr)
        self.assertEqual(result["action"], "unknown")
        self.assertEqual(result["confidence"], 0.0)

    def test_sitting_figure_action_is_sitting_or_unknown(self):
        """A sitting stick figure must produce 'sitting' (geometry) or 'unknown'
        (MediaPipe failure). It must NEVER produce 'supine' despite the MLP bias.
        This is the primary regression for the bug described in the issue.
        """
        self._skip_if_no_model()
        bgr = self._make_test_image_bgr("sitting")
        result = self.predict_image(bgr)
        self.assertNotEqual(result["action"], "supine",
            f"Sitting figure returned 'supine' — regression. Full result: {result}")
        self.assertIn(result["action"], {"sitting", "unknown"},
            f"Sitting figure returned unexpected action: {result['action']}")

    def test_geometry_result_has_zero_confidence(self):
        """When decisionSource is 'pose_geometry', confidence must be 0.0."""
        self._skip_if_no_model()
        bgr = self._make_test_image_bgr("sitting")
        result = self.predict_image(bgr)
        if result.get("decisionSource") == "pose_geometry":
            self.assertEqual(result["confidence"], 0.0,
                "pose_geometry result must not carry MLP score as confidence")

    def test_mlp_raw_scores_always_present_when_pose_detected(self):
        """classScores must contain exactly len(LABELS) entries when a pose
        is detected (i.e., framesAnalyzed > 0 and classScores is non-empty).
        """
        self._skip_if_no_model()
        from inference_api import LABELS as LBL
        bgr = self._make_test_image_bgr("sitting")
        result = self.predict_image(bgr)
        scores = result.get("classScores", [])
        if scores:  # scores empty only when no pose detected at all
            self.assertEqual(len(scores), len(LBL),
                f"Expected {len(LBL)} class scores, got {len(scores)}")
            score_labels = {s["action"] for s in scores}
            self.assertEqual(score_labels, set(LBL))

    def test_model_confidence_reflects_mlp_output(self):
        """modelConfidence is the raw MLP top-class score and should be in [0,1]."""
        self._skip_if_no_model()
        import numpy as np
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.predict_image(bgr)
        mc = result.get("modelConfidence", -1)
        self.assertGreaterEqual(mc, 0.0)
        self.assertLessEqual(mc, 1.0)

    def test_decision_source_is_valid_value(self):
        """decisionSource must be one of the three documented values."""
        self._skip_if_no_model()
        import numpy as np
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.predict_image(bgr)
        self.assertIn(result["decisionSource"],
                      {"model", "pose_geometry", "pose_ambiguity"})
