import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xml.etree.ElementTree as ET

#matplotlib.use("TkAgg")
#plt.ion()
os.environ["MUJOCO_GL"] = "egl"

class SquatPDOnlyEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        #self.renderer = mujoco.Renderer(self.model, 640, 480)

        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.num_ctrl = self.model.nu

        self.standing_qpos = self._make_standing_pose()
        self.squat_qpos = self._make_squat_pose()

        self.abdomen_y_id = self.model.joint("abdomen_y").id
        self.abdomen_x_id = self.model.joint("abdomen_x").id
        
        self.imu_gyro_id = self.model.sensor("imu_gyro").id
        self.imu_accel_id = self.model.sensor("imu_acc").id
        self.imu_site_id = self.model.site("imu_site").id
        self.velocity_id = self.model.sensor("velocity").id
        self.dt = self.model.opt.timestep
        self.prev_vel = np.zeros(3)
        self.vel = np.zeros((6, 1))
        #print(self.imu_gyro_id, self.imu_accel_id)
        
        #phone, watch position
        self.phone_pos_id = self.model.sensor("watch_position").id
        self.watch_pos_id = self.model.sensor("phone_position").id
        #print(self.phone_pos_id, self.watch_pos_id)

        self.data.qpos[:] = self.standing_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

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

    def _make_standing_pose(self):
        return np.array([
            0, 0, 1.25,  # torso 위치
            1, 0, 0, 0,  # torso 방향
            0, 0, 0,  # 척추 회전

            0, 0, 0, 0, 0, 0,  # 오른쪽 다리
            0, 0, 0, 0, 0, 0,  # 왼쪽 다리

            0, 0, 0, 0, 0, 0  # 팔
        ])

    def run_pd_squat(self, down_qpos_th, squat_data:list, up_qpos_th = 1.0, max_frames=200):
            for phase in ["down", "up"]:
                print(f"Phase {phase}")
                target = self.squat_qpos if phase == "down" else self.standing_qpos

                for frame in range(max_frames):
                    qpos_err = np.linalg.norm(self.data.qpos - target)
                    print(f"Frame {frame} - Qpos error: {qpos_err:.3f}")
                    if phase == "down" and qpos_err < down_qpos_th:
                        #print(f"Phase {phase} completed at frame {frame}")
                        break
                    if phase == "up" and qpos_err < up_qpos_th:
                        #print(f"Phase {phase} completed at frame {frame}")
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

                    if phase == "up" and frame % 10 == 0:
                        com = self.data.subtree_com[0]
                        print(f"Frame {frame} - COM x: {com[0]:.3f}")

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

    down_qpos_th = np.random.uniform(0.8, 1.0)

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
    return output_xml ,down_qpos_th

if __name__ == "__main__":
    squat_data = []
    original_xml = "humanoid_WSensor.xml"
    modified_xml = "humanoid_WSensor_modified.xml"
    #env = SquatPDOnlyEnv(xml_path)
    for cycle in range(0, 3001):
        modified_xml, down_qpos_th = set_random_position(original_xml, modified_xml)
        env = SquatPDOnlyEnv(modified_xml)
        env.run_pd_squat(squat_data = squat_data, down_qpos_th = down_qpos_th)
        #env.close()
    df = pd.DataFrame(squat_data)
    df.to_csv("squat_data.csv", index=False)
    
