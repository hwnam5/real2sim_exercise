import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib
import xml.etree.ElementTree as ET
import pandas as pd

#matplotlib.use("TkAgg")
#plt.ion()
os.environ["MUJOCO_GL"] = "egl"

class SitUpPDEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        #self.renderer = mujoco.Renderer(self.model, 640, 480)

        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.num_ctrl = self.model.nu

        self.situp_down_qpos = self._make_situp_down_pose()
        self.situp_up_qpos = self._make_situp_up_pose()

        self.abdomen_y_id = self.model.joint("abdomen_y").id
        self.abdomen_x_id = self.model.joint("abdomen_x").id
        
        self.imu_gyro_id = self.model.sensor("imu_gyro").id
        self.imu_accel_id = self.model.sensor("imu_acc").id
        self.imu_site_id = self.model.site("imu_site").id
        self.velocity_id = self.model.sensor("velocity").id

        self.phone_pos_id = self.model.sensor("watch_position").id
        self.watch_pos_id = self.model.sensor("phone_position").id

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


    def run_pd_situp_once(self, up_qpos_th, squat_data:list, max_frames=200):
        for phase in ["down", "up", "down"]:
            print(f"Phase: {phase}")
            target = self.situp_up_qpos if phase == "up" else self.situp_down_qpos

            for frame in range(max_frames):
                qpos_err = np.linalg.norm(self.data.qpos - target)
                print(f"Frame {frame} - Qpos error: {qpos_err:.3f}")
                if qpos_err < up_qpos_th and phase == "up":
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
                
                phone_pos_data = self.data.sensordata[self.phone_pos_id * 3:self.phone_pos_id*3 + 3]
                watch_pos_data = self.data.sensordata[self.watch_pos_id * 3:self.watch_pos_id*3 + 3]
                distance = np.linalg.norm(phone_pos_data - watch_pos_data)

                #accel
                lin_vel = self.data.sensordata[self.velocity_id * 3:self.velocity_id*3 + 3]
                accel_est = self.data.sensordata[self.imu_accel_id * 3:self.imu_accel_id*3 + 3]
                ang_vel = self.data.sensordata[self.imu_gyro_id * 3:self.imu_gyro_id*3 + 3]

                squat_data.append({
                    "updown" : 1 if phase == "up" else 0,
                    "UWB" : distance,
                    "vel_x" : lin_vel[0],
                    "vel_y" : lin_vel[1],
                    "vel_z" : lin_vel[2],
                    "accel_x" : accel_est[0],
                    "accel_y" : accel_est[1],
                    "accel_z" : accel_est[2],
                    "gyro_x" : ang_vel[0],
                    "gyro_y" : ang_vel[1],
                    "gyro_z" : ang_vel[2],
                })

                #self.renderer.update_scene(self.data)
                #img = self.renderer.render()
                #plt.imshow(img)
                #plt.axis("off")
                #plt.pause(0.01)

        #plt.close()

    def close(self):
        self.renderer.close()


def set_random_position(xml_path, output_xml):

    phone_x = np.random.uniform(0.2, 1.5)
    phone_y = np.random.uniform(-1.5, 1.5)
    phone_z = np.random.uniform(0, 0.7)

    imu_x = np.random.uniform(-0.05, 0.05)
    imu_y = np.random.uniform(-0.05, 0.05)
    imu_z = np.random.uniform(-0.05, 0.05)

    up_qpos_th = np.random.uniform(0.8, 1.0)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is not None:
        for body in worldbody.findall("body"):
            if body.get("name") == "phone":
                body.set("pos", f"{phone_x} {phone_y} {phone_z}")
                break
    for site in root.iter("site"):
        if site.get("name") == "imu_site":
            site.set("pos", f"{imu_x} {imu_y} {imu_z}")
            break
        

    tree.write(output_xml)
    return output_xml ,up_qpos_th

if __name__ == "__main__":#
    squat_data = []
    original_xml = "humanoid_WSensor.xml"
    modified_xml = "humanoid_WSensor_modified.xml"
    #env = SquatPDOnlyEnv(xml_path)
    for cycle in range(0, 1000):
        modified_xml, up_qpos_th = set_random_position(original_xml, modified_xml)
        env = SitUpPDEnv(modified_xml)
        env.run_pd_situp_once(squat_data = squat_data, up_qpos_th = up_qpos_th)
        #env.close()
    df = pd.DataFrame(squat_data)
    df.to_csv("situp_data.csv", index=False)
