import unittest
from types import SimpleNamespace

from inference_api import (
    all_fours_support_from_landmarks,
    body_pose_in_frame,
    prone_support_from_landmarks,
    recumbent_pose_is_ambiguous,
    select_pose_geometry_action,
    upright_posture_from_landmarks,
)


def pose(*, head, shoulders, hips, knees, ankles, wrists=(0.5, 0.5), elbows=(0.5, 0.5)):
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]

    def set_pair(indices, center, spread=0.01):
        landmarks[indices[0]] = SimpleNamespace(x=center[0] - spread, y=center[1], visibility=1.0)
        landmarks[indices[1]] = SimpleNamespace(x=center[0] + spread, y=center[1], visibility=1.0)

    landmarks[0] = SimpleNamespace(x=head[0], y=head[1], visibility=1.0)
    set_pair((11, 12), shoulders)
    set_pair((23, 24), hips)
    set_pair((25, 26), knees, spread=0.015)
    set_pair((27, 28), ankles, spread=0.02)
    set_pair((13, 14), elbows, spread=0.015)
    set_pair((15, 16), wrists)
    return landmarks


class UprightPostureGeometryTests(unittest.TestCase):
    def test_clear_standing_pose(self):
        landmarks = pose(
            head=(0.5, 0.14), shoulders=(0.5, 0.31), hips=(0.5, 0.52),
            knees=(0.5, 0.70), ankles=(0.5, 0.91),
        )
        self.assertEqual(upright_posture_from_landmarks(landmarks), "standing")
        self.assertIsNone(all_fours_support_from_landmarks(landmarks))

    def test_clear_upright_geometry_wins_over_an_all_fours_candidate(self):
        self.assertEqual(
            select_pose_geometry_action("supine", "standing", "all-fours"),
            "standing",
        )
        self.assertEqual(
            select_pose_geometry_action("supine", "sitting", "all-fours"),
            "sitting",
        )

    def test_all_fours_shape_can_correct_a_back_guess_when_no_other_shape_matches(self):
        self.assertEqual(
            select_pose_geometry_action("supine", None, "all-fours"),
            "all-fours",
        )

    def test_all_fours_fallback_does_not_replace_learned_standing_or_sitting(self):
        for upright_action in ("standing", "sitting"):
            with self.subTest(action=upright_action):
                self.assertIsNone(
                    select_pose_geometry_action(upright_action, "all-fours", "all-fours")
                )

    def test_clear_sitting_pose_with_forward_thighs(self):
        landmarks = pose(
            head=(0.5, 0.15), shoulders=(0.5, 0.33), hips=(0.5, 0.61),
            knees=(0.62, 0.61), ankles=(0.77, 0.78),
        )
        self.assertEqual(upright_posture_from_landmarks(landmarks), "sitting")

    def test_cropped_or_asymmetric_leg_does_not_create_a_geometry_sitting_result(self):
        landmarks = pose(
            head=(0.593, 0.317), shoulders=(0.566, 0.475), hips=(0.554, 0.803),
            knees=(0.615, 0.874), ankles=(0.648, 0.907), wrists=(0.64, 0.74),
            elbows=(0.613, 0.684),
        )
        landmarks[23].x, landmarks[23].y = 0.614, 0.781
        landmarks[24].x, landmarks[24].y = 0.494, 0.824
        landmarks[25].x, landmarks[25].y, landmarks[25].visibility = 0.722, 0.732, 0.1
        landmarks[26].x, landmarks[26].y, landmarks[26].visibility = 0.507, 0.874, 0.52
        self.assertIsNone(upright_posture_from_landmarks(landmarks))
        self.assertEqual(all_fours_support_from_landmarks(landmarks), "all-fours")

    def test_tummy_time_with_knees_above_hips_is_not_mislabeled_sitting(self):
        landmarks = pose(
            head=(0.5, 0.20), shoulders=(0.5, 0.40), hips=(0.5, 0.69),
            knees=(0.58, 0.63), ankles=(0.63, 0.62), wrists=(0.64, 0.83),
        )
        self.assertIsNone(upright_posture_from_landmarks(landmarks))
        self.assertEqual(prone_support_from_landmarks(landmarks), "prone")

    def test_supported_tummy_time_with_head_just_above_the_shoulders(self):
        landmarks = pose(
            head=(0.5, 0.37), shoulders=(0.5, 0.40), hips=(0.5, 0.69),
            knees=(0.58, 0.63), ankles=(0.63, 0.62), wrists=(0.64, 0.83),
        )
        self.assertEqual(prone_support_from_landmarks(landmarks), "prone")

    def test_clear_hands_and_knees_pose_does_not_need_feet_in_frame(self):
        landmarks = pose(
            head=(0.593, 0.317), shoulders=(0.566, 0.475), hips=(0.554, 0.803),
            knees=(0.615, 0.874), ankles=(0.648, 0.907), wrists=(0.64, 0.74),
            elbows=(0.613, 0.684),
        )
        landmarks[23].x, landmarks[23].y = 0.614, 0.781
        landmarks[24].x, landmarks[24].y = 0.494, 0.824
        landmarks[25].x, landmarks[25].y, landmarks[25].visibility = 0.722, 0.732, 0.1
        landmarks[26].x, landmarks[26].y, landmarks[26].visibility = 0.507, 0.874, 0.52
        for index in (27, 28):
            landmarks[index].y = 1.2
            landmarks[index].visibility = 0.1
        self.assertFalse(body_pose_in_frame(landmarks))
        self.assertEqual(all_fours_support_from_landmarks(landmarks), "all-fours")

    def test_all_fours_rejects_a_single_weak_landmark_in_each_required_pair(self):
        for index in (11, 12, 13, 14, 15, 16, 23, 24):
            with self.subTest(index=index):
                landmarks = pose(
                    head=(0.593, 0.317), shoulders=(0.566, 0.475), hips=(0.554, 0.803),
                    knees=(0.615, 0.874), ankles=(0.648, 0.907), wrists=(0.64, 0.74),
                    elbows=(0.613, 0.684),
                )
                landmarks[index].visibility = 0.22
                self.assertIsNone(all_fours_support_from_landmarks(landmarks))

    def test_all_fours_requires_a_clear_head_and_one_confident_support_knee(self):
        landmarks = pose(
            head=(0.593, 0.317), shoulders=(0.566, 0.475), hips=(0.554, 0.803),
            knees=(0.615, 0.874), ankles=(0.648, 0.907), wrists=(0.64, 0.74),
            elbows=(0.613, 0.684),
        )
        landmarks[0].visibility = 0.28
        self.assertIsNone(all_fours_support_from_landmarks(landmarks))

        landmarks[0].visibility = 1.0
        landmarks[25].visibility = 0.24
        landmarks[26].visibility = 0.24
        self.assertIsNone(all_fours_support_from_landmarks(landmarks))

    def test_tummy_time_and_back_lying_are_not_mistaken_for_all_fours(self):
        tummy_time = pose(
            head=(0.5, 0.37), shoulders=(0.5, 0.40), hips=(0.5, 0.69),
            knees=(0.58, 0.63), ankles=(0.63, 0.62), wrists=(0.64, 0.83),
            elbows=(0.60, 0.68),
        )
        supine = pose(
            head=(0.5, 0.40), shoulders=(0.5, 0.43), hips=(0.5, 0.70),
            knees=(0.5, 0.71), ankles=(0.55, 0.63), wrists=(0.5, 0.79),
            elbows=(0.5, 0.61),
        )
        self.assertIsNone(all_fours_support_from_landmarks(tummy_time))
        self.assertIsNone(all_fours_support_from_landmarks(supine))

    def test_supine_pose_without_supported_tummy_pattern_is_not_overridden(self):
        landmarks = pose(
            head=(0.5, 0.40), shoulders=(0.5, 0.43), hips=(0.5, 0.70),
            knees=(0.5, 0.71), ankles=(0.55, 0.63), wrists=(0.5, 0.79),
        )
        self.assertIsNone(prone_support_from_landmarks(landmarks))
        self.assertTrue(recumbent_pose_is_ambiguous(landmarks))

    def test_horizontal_laying_pose_is_ambiguous_without_tummy_support_cue(self):
        landmarks = pose(
            head=(0.10, 0.42), shoulders=(0.30, 0.44), hips=(0.58, 0.49),
            knees=(0.74, 0.54), ankles=(0.91, 0.55), wrists=(0.42, 0.68),
        )
        self.assertIsNone(prone_support_from_landmarks(landmarks))
        self.assertTrue(recumbent_pose_is_ambiguous(landmarks))

    def test_clear_standing_head_to_shoulder_direction_is_not_ambiguous(self):
        landmarks = pose(
            head=(0.5, 0.14), shoulders=(0.5, 0.31), hips=(0.5, 0.52),
            knees=(0.5, 0.70), ankles=(0.5, 0.91),
        )
        self.assertFalse(recumbent_pose_is_ambiguous(landmarks))

    def test_recumbent_pose_is_left_to_the_learned_model(self):
        landmarks = pose(
            head=(0.2, 0.48), shoulders=(0.34, 0.50), hips=(0.59, 0.52),
            knees=(0.67, 0.60), ankles=(0.75, 0.61),
        )
        self.assertIsNone(upright_posture_from_landmarks(landmarks))

    def test_missing_pose_landmarks_abstains(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
        self.assertIsNone(upright_posture_from_landmarks(landmarks))
        self.assertFalse(body_pose_in_frame(landmarks))
        self.assertFalse(recumbent_pose_is_ambiguous(landmarks))

    def test_in_frame_gate_requires_head_and_arm_evidence(self):
        without_head = pose(
            head=(0.5, 0.14), shoulders=(0.5, 0.31), hips=(0.5, 0.52),
            knees=(0.5, 0.70), ankles=(0.5, 0.91),
        )
        without_head[0].visibility = 0.1
        without_arms = pose(
            head=(0.5, 0.14), shoulders=(0.5, 0.31), hips=(0.5, 0.52),
            knees=(0.5, 0.70), ankles=(0.5, 0.91),
        )
        for index in (13, 14, 15, 16):
            without_arms[index].visibility = 0.1
        self.assertFalse(body_pose_in_frame(without_head))
        self.assertFalse(body_pose_in_frame(without_arms))

    def test_upright_geometry_requires_both_knee_and_ankle_chains(self):
        landmarks = pose(
            head=(0.5, 0.14), shoulders=(0.5, 0.31), hips=(0.5, 0.52),
            knees=(0.5, 0.70), ankles=(0.5, 0.91),
        )
        landmarks[26].visibility = 0.1
        self.assertIsNone(upright_posture_from_landmarks(landmarks))

    def test_pose_with_most_lower_body_joints_out_of_frame_is_rejected(self):
        landmarks = pose(
            head=(0.5, 0.14), shoulders=(0.5, 0.31), hips=(1.4, 0.52),
            knees=(1.5, 0.70), ankles=(1.6, 0.91),
        )
        self.assertFalse(body_pose_in_frame(landmarks))


if __name__ == "__main__":
    unittest.main()
