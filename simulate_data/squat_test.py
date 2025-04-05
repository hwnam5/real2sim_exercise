import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
plt.ion()
os.environ["MUJOCO_GL"] = "egl"

class SquatPDOnlyEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 640, 480)

        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.num_ctrl = self.model.nu

        self.standing_qpos = self._make_standing_pose()
        self.squat_qpos = self._make_squat_pose()

        self.abdomen_y_id = self.model.joint("abdomen_y").id
        self.abdomen_x_id = self.model.joint("abdomen_x").id

        self.data.qpos[:] = self.standing_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _make_standing_pose(self):
        return np.array([
            0, 0, 1.25,  # torso 위치
            1, 0, 0, 0,  # torso 방향
            0, 0, 0,  # 척추 회전

            0, 0, 0, 0, 0, 0,  # 오른쪽 다리
            0, 0, 0, 0, 0, 0,  # 왼쪽 다리

            0, 0, 0, 0, 0, 0  # 팔
        ])

    def _make_squat_pose(self):
        def r(low, high):  # 간단한 helper
            return np.random.uniform(low, high)

        return np.array([
            0 + r(-0.02, 0.02), 0 + r(-0.02, 0.02), 0.75 + r(-0.01, 0.01),  # torso pos
            1, 0, 0, 0,  # no randomness in quaternion
            0 + r(-0.03, 0.03), 0.4 + r(-0.05, 0.05), 0,  # abdomen_z/y/x
            -0.25 + r(-0.05, 0.05), -0.4 + r(-0.05, 0.05), -2.0 + r(-0.1, 0.1), -2.3 + r(-0.1, 0.1), -0.6 + r(-0.05, 0.05), 0.4 + r(-0.05, 0.05),  # right leg
            -0.25 + r(-0.05, 0.05), -0.4 + r(-0.05, 0.05), -2.0 + r(-0.1, 0.1), -2.3 + r(-0.1, 0.1), -0.6 + r(-0.05, 0.05), 0.4 + r(-0.05, 0.05),  # left leg
            0, 0, 0, 0, 0, 0  # 팔은 고정
        ])
    def run_pd_squat(self, cycles=3, max_frames=200):
        for cycle in range(cycles):
            for phase in ["down", "up"]:
                print(f"Cycle {cycle + 1}, Phase {phase}")
                target = self.squat_qpos if phase == "down" else self.standing_qpos

                for frame in range(max_frames):
                    qpos_err = np.linalg.norm(self.data.qpos - target)
                    print(f"Frame {frame} - Qpos error: {qpos_err:.3f}")
                    if phase == "down" and qpos_err < 0.9:
                        print(f"Cycle {cycle + 1}, Phase {phase} completed at frame {frame}")
                        break
                    if phase == "up" and qpos_err < 1.1:
                        print(f"Cycle {cycle + 1}, Phase {phase} completed at frame {frame}")
                        break

                    com_x = self.data.subtree_com[0][0]
                    feet_x = self.data.qpos[0]
                    delta_x = com_x - feet_x

                    for k in range(self.model.nu):
                        joint_id = self.model.actuator_trnid[k][0]
                        qpos_i = self.model.jnt_qposadr[joint_id]
                        qvel_i = self.model.jnt_dofadr[joint_id]
                        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

                        err = target[qpos_i] - self.data.qpos[qpos_i]
                        derr = -self.data.qvel[qvel_i]
                        ctrl = 20.0 * err + 3.0 * derr

                        # 1. 상체 고정 (업 phase에 더 강하게)
                        if joint_name in ["abdomen_y", "abdomen_x", "abdomen_z"]:
                            if phase == "up":
                                upright_target = 0.0
                                upright_err = upright_target - self.data.qpos[qpos_i]
                                upright_d = -self.data.qvel[qvel_i]
                                ctrl += 80.0 * upright_err + 10.0 * upright_d

                        # 2. 팔 관절 고정
                        elif "shoulder" in joint_name or "elbow" in joint_name or "wrist" in joint_name:
                            ctrl = 40.0 * err + 5.0 * derr

                        # 3. 발목은 따로 balance 제어
                        elif joint_name in ["ankle_y_right", "ankle_y_left"]:
                            foot_qvel = self.data.qvel[qvel_i]
                            foot_ctrl = 2.0 * delta_x - 0.5 * foot_qvel
                            ctrl = np.clip(foot_ctrl, -0.3, 0.3)

                        self.data.ctrl[k] = ctrl

                    mujoco.mj_step(self.model, self.data)

                    if phase == "up" and frame % 10 == 0:
                        com = self.data.subtree_com[0]
                        print(f"[Cycle {cycle + 1}] Frame {frame} - COM x: {com[0]:.3f}")

                    self.renderer.update_scene(self.data)
                    img = self.renderer.render()
                    plt.imshow(img)
                    plt.axis("off")
                    plt.pause(0.01)
        plt.close()

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    xml_path = "humanoid_WSensor_modified.xml"
    env = SquatPDOnlyEnv(xml_path)
    env.run_pd_squat(cycles=3)
    env.close()