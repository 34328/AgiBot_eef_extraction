from enum import Enum
from typing import Dict, List, Optional

import numpy as np

try:
    import pinocchio as pin
except Exception as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "ik_solver.py requires pinocchio. Install pinocchio to use this solver."
    ) from exc

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class RobotPart(Enum):
    LEFT_ARM = 0
    RIGHT_ARM = 1


class Solver:
    def __init__(
        self,
        config_path: str,
        urdf_path: str,
        use_relaxed_ik: bool = True,
        use_elbow: bool = False,
    ) -> None:
        # NOTE: use_elbow is accepted for API parity but not implemented here.
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._q = pin.neutral(self._model).copy()

        self._base_frame = self._model.getFrameId("base_link")
        self._center_frame = self._model.getFrameId("arm_base_link")
        self._left_ee_frame = self._model.getFrameId("arm_left_link7")
        self._right_ee_frame = self._model.getFrameId("arm_right_link7")

        self._left_joints = [
            "idx05_left_arm_joint1",
            "idx06_left_arm_joint2",
            "idx07_left_arm_joint3",
            "idx08_left_arm_joint4",
            "idx09_left_arm_joint5",
            "idx10_left_arm_joint6",
            "idx11_left_arm_joint7",
        ]
        self._right_joints = [
            "idx12_right_arm_joint1",
            "idx13_right_arm_joint2",
            "idx14_right_arm_joint3",
            "idx15_right_arm_joint4",
            "idx16_right_arm_joint5",
            "idx17_right_arm_joint6",
            "idx18_right_arm_joint7",
        ]
        self._head_joints = [
            "idx03_head_yaw_joint",
            "idx04_head_pitch_joint",
        ]

        self._joint_q_index = self._build_joint_q_index()
        self._joint_v_index = self._build_joint_v_index()

        self._target: Dict[RobotPart, Optional[pin.SE3]] = {
            RobotPart.LEFT_ARM: None,
            RobotPart.RIGHT_ARM: None,
        }
        self._debug = False

        self._cfg = self._load_config(config_path)
        if use_relaxed_ik:
            self._ik_cfg = self._cfg.get("relaxed_ik", {})
        else:
            self._ik_cfg = self._cfg.get("standard_ik", {})
        self._opt_cfg = self._cfg.get("optimizer", {})

    def _build_joint_q_index(self) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        for name in self._left_joints + self._right_joints + self._head_joints:
            jid = self._model.getJointId(name)
            indices[name] = self._model.joints[jid].idx_q
        return indices

    def _build_joint_v_index(self) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        for name in self._left_joints + self._right_joints + self._head_joints:
            jid = self._model.getJointId(name)
            indices[name] = self._model.joints[jid].idx_v
        return indices

    def _load_config(self, config_path: str) -> Dict[str, Dict[str, float]]:
        if not config_path or yaml is None:
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _set_joint_positions(self, q: np.ndarray, joint_names: List[str], values: np.ndarray) -> None:
        for name, val in zip(joint_names, values):
            q_idx = self._joint_q_index[name]
            q[q_idx] = float(val)

    def initialize_states(
        self,
        left_arm_init: np.ndarray,
        right_arm_init: np.ndarray,
        head_init: np.ndarray,
    ) -> None:
        self._q = pin.neutral(self._model).copy()
        self._set_joint_positions(self._q, self._left_joints, left_arm_init)
        self._set_joint_positions(self._q, self._right_joints, right_arm_init)
        self._set_joint_positions(self._q, self._head_joints, head_init)

    def set_debug_mode(self, enabled: bool) -> None:
        self._debug = bool(enabled)

    def compute_fk(self, q: np.ndarray, start_link: str, end_link: str) -> np.ndarray:
        q_full = np.array(q, dtype=np.float64, copy=True)
        pin.forwardKinematics(self._model, self._data, q_full)
        pin.updateFramePlacements(self._model, self._data)
        start_id = self._model.getFrameId(start_link)
        end_id = self._model.getFrameId(end_link)
        oMs = self._data.oMf[start_id]
        oMe = self._data.oMf[end_id]
        T = oMs.inverse() * oMe
        return T.homogeneous

    def compute_part_fk(
        self,
        q_part: np.ndarray,
        part: RobotPart,
        from_base: bool = False,
    ) -> np.ndarray:
        q_full = pin.neutral(self._model).copy()
        if part == RobotPart.LEFT_ARM:
            self._set_joint_positions(q_full, self._left_joints, q_part)
            ee_frame = self._left_ee_frame
        else:
            self._set_joint_positions(q_full, self._right_joints, q_part)
            ee_frame = self._right_ee_frame

        pin.forwardKinematics(self._model, self._data, q_full)
        pin.updateFramePlacements(self._model, self._data)
        oMee = self._data.oMf[ee_frame]

        if from_base:
            return oMee.homogeneous

        oMcenter = self._data.oMf[self._center_frame]
        centerMee = oMcenter.inverse() * oMee
        return centerMee.homogeneous

    def update_target_mat(self, part: RobotPart, target_pos: np.ndarray, target_rot: np.ndarray) -> None:
        self._target[part] = pin.SE3(target_rot, target_pos)

    def solve_left_arm(self) -> np.ndarray:
        return self._solve_arm(RobotPart.LEFT_ARM)

    def solve_right_arm(self) -> np.ndarray:
        return self._solve_arm(RobotPart.RIGHT_ARM)

    def _solve_arm(self, part: RobotPart) -> np.ndarray:
        target = self._target.get(part)
        if target is None:
            raise RuntimeError("Target pose not set for IK solver.")

        q = self._q.copy()
        if part == RobotPart.LEFT_ARM:
            joint_names = self._left_joints
            ee_frame = self._left_ee_frame
        else:
            joint_names = self._right_joints
            ee_frame = self._right_ee_frame

        v_idx = [self._joint_v_index[n] for n in joint_names]
        q_idx = [self._joint_q_index[n] for n in joint_names]

        max_iter = int(self._opt_cfg.get("maxeval", 100))
        eps = float(self._opt_cfg.get("xtol_rel", 1e-4))
        damping = float(self._opt_cfg.get("initial_step", 1e-4))
        pos_w = float(self._ik_cfg.get("pos_weight", 1.0))
        ori_w = float(self._ik_cfg.get("ori_weight", 1.0))

        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)

            oMcenter = self._data.oMf[self._center_frame]
            oMee = self._data.oMf[ee_frame]
            centerMee = oMcenter.inverse() * oMee

            err_se3 = pin.log6(centerMee.inverse() * target).vector
            # pin.log6 returns [linear, angular]
            err_se3[:3] *= pos_w
            err_se3[3:] *= ori_w
            if np.linalg.norm(err_se3) < eps:
                break

            J = pin.computeFrameJacobian(
                self._model, self._data, q, ee_frame, pin.ReferenceFrame.LOCAL
            )
            J_arm = J[:, v_idx]
            J_arm[:3, :] *= pos_w
            J_arm[3:, :] *= ori_w

            # Damped least squares.
            JJt = J_arm @ J_arm.T
            dq = J_arm.T @ np.linalg.solve(JJt + damping * np.eye(6), err_se3)

            for i, qi in enumerate(q_idx):
                q[qi] += dq[i]

        self._q = q
        return np.array([q[i] for i in q_idx], dtype=np.float32)
