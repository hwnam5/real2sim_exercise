import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
plt.ion()
os.environ["MUJOCO_GL"] = "egl"

class SitUpPDEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 640, 480)

        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.num_ctrl = self.model.nu

        self.situp_down_qpos = self._make_situp_down_pose()
        self.situp_up_qpos = self._make_situp_up_pose()

        self.data.qpos[:] = self.situp_down_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _make_situp_down_pose(self):
        def r(low, high):
            return np.random.uniform(low, high)

        return np.array([
            -0.4 + r(-0.01, 0.01), 0 + r(-0.01, 0.01), 0.09122 + r(-0.01, 0.01),
            0.322788 + r(-0.01, 0.01), 0 + r(-0.01, 0.01), -0.19107 + r(-0.01, 0.01), 0,
            0 + r(-0.02, 0.02), -1.5 + r(-0.02, 0.02), 0 + r(-0.02, 0.02),
            0.0182 + r(-0.01, 0.01), 0.0142 + r(-0.01, 0.01), 0.3 + r(-0.05, 0.05), -1.9 + r(-0.05, 0.05), -0.44 + r(-0.01, 0.01), -0.02 + r(-0.01, 0.01),
            0.0182 + r(-0.01, 0.01), 0.0142 + r(-0.01, 0.01), 0.3 + r(-0.05, 0.05), -1.9 + r(-0.05, 0.05), -0.44 + r(-0.01, 0.01), -0.02 + r(-0.01, 0.01),
            -0.1 + r(-0.02, 0.02), 0.2 + r(-0.02, 0.02), 0.7 + r(-0.02, 0.02),
            -0.1 + r(-0.02, 0.02), 0.2 + r(-0.02, 0.02), 0.7 + r(-0.02, 0.02)
        ])

    def _make_situp_up_pose(self):
        def r(low, high):
            return np.random.uniform(low, high)

        return np.array([
            -0.4 + r(-0.01, 0.01), 0 + r(-0.01, 0.01), 0.39122 + r(-0.01, 0.01),   # torso 위치
            0.322788 + r(-0.01, 0.01), 0 + r(-0.01, 0.01), 0.09107 + r(-0.01, 0.01), 0,  # orientation (quat)
            0 + r(-0.02, 0.02), -1.5 + r(-0.02, 0.02), 0 + r(-0.02, 0.02),               # 허리
            0.0182 + r(-0.01, 0.01), 0.0142 + r(-0.01, 0.01), -1.5 + r(-0.05, 0.05), -1.9 + r(-0.05, 0.05), -0.44 + r(-0.01, 0.01), -0.02 + r(-0.01, 0.01),  # 왼다리
            0.0182 + r(-0.01, 0.01), 0.0142 + r(-0.01, 0.01), -1.5 + r(-0.05, 0.05), -1.9 + r(-0.05, 0.05), -0.44 + r(-0.01, 0.01), -0.02 + r(-0.01, 0.01),  # 오른다리
            -0.1 + r(-0.02, 0.02), 0.2 + r(-0.02, 0.02), 0.7 + r(-0.02, 0.02),   # 왼팔
            -0.1 + r(-0.02, 0.02), 0.2 + r(-0.02, 0.02), 0.7 + r(-0.02, 0.02)    # 오른팔
        ])

    def run_pd_situp_once(self, max_frames=200):
        for phase in ["down", "up", "down"]:
            print(f"Phase: {phase}")
            target = self.situp_up_qpos if phase == "up" else self.situp_down_qpos

            for frame in range(max_frames):
                qpos_err = np.linalg.norm(self.data.qpos - target)
                print(f"Frame {frame} - Qpos error: {qpos_err:.3f}")
                if qpos_err < 0.9 and phase == "up":
                    print(f"Phase {phase} completed at frame {frame}")
                    break
                elif qpos_err < 1.1 and phase == "down":
                    print(f"Phase {phase} completed at frame {frame}")
                    break

                for k in range(self.model.nu):
                    joint_id = self.model.actuator_trnid[k][0]
                    qpos_i = self.model.jnt_qposadr[joint_id]
                    qvel_i = self.model.jnt_dofadr[joint_id]
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

                    err = target[qpos_i] - self.data.qpos[qpos_i]
                    derr = -self.data.qvel[qvel_i]

                    if joint_name in ["abdomen_y", "abdomen_x", "abdomen_z"]:
                        ctrl = 40.0 * err + 6.0 * derr
                    else:
                        ctrl = 20.0 * err + 3.0 * derr

                    self.data.ctrl[k] = ctrl

                mujoco.mj_step(self.model, self.data)

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
    env = SitUpPDEnv(xml_path)
    env.run_pd_situp_once()
    env.close()
