import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mujoco

# matplotlib 설정 (TkAgg 백엔드 사용)
matplotlib.use("TkAgg")
plt.ion()
os.environ["MUJOCO_GL"] = "egl"  # 또는 "osmesa", "glfw" 등 시스템에 맞게 조정

def view_single_squat_pose(xml_path, squat_qpos):
    # 모델과 데이터 로딩
    model = mujoco.MjModel.from_xml_path(xml_path)
    for i in range(model.njnt):
        print(f"joint {i}: name={model.joint(i).name}, qposadr={model.jnt_qposadr[i]}")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 1280, 1280)

    # 스쿼트 자세 적용
    data.qpos[:] = squat_qpos
    mujoco.mj_forward(model, data)

    # 일정 시간 동안 렌더링 반복
    for _ in range(300):  # 약 9초 동안 표시 (0.03초 * 300)
        renderer.update_scene(data)
        img = renderer.render()
        plt.imshow(img)
        plt.axis("off")
        plt.pause(0.03)
    plt.show()

if __name__ == "__main__":
    xml_path = "humanoid_WSensor.xml"  # ← 본인의 xml 경로로 수정하세요
    standing_qpos = np.array([
            0, 0, 1.25,  # torso 위치
            1, 0, 0, 0,  # torso 방향
            0, 0, 0,  # 척추 회전

            0, 0, 0, 0, 0, 0,  # 오른쪽 다리
            0, 0, 0, 0, 0, 0,  # 왼쪽 다리

            0, 0, 0, 0, 0, 0  # 팔
        ])
    # squat_qpos 정의 (길이: model.nq와 맞아야 함)
    squat_qpos = np.array([
        0, 0, 0.93,  # torso z 약간 높임
        0.995, 0, 0, 0,  # orientation 약간만 기울임
        0, 0.5, 0,  # abdomen 조정

        # 오른쪽 다리 (10~15)
        -0.25, -0.4, -1.6, -1.5, -0.4, 0.4,

        # 왼쪽 다리 (16~21)
        -0.25, -0.4, -1.6, -1.5, -0.4, 0.4,

        # 팔 (22~27)
        0, 0, 0, 0, 0, 0
    ])
    original_squat_qpos = np.array([
        0, 0, 0.596,
        0.988015, 0, 0.154359, 0,
        0, 0.4, 0,
        -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
        -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
        0, 0, 0, 0, 0, 0
    ])
    raised_squat_qpos = np.array([
            0, 0, 0.75,
            1, 0, 0, 0,
            0, 0.4, 0,
            -0.25, -0.4, -2.0, -2.3, -0.6, 0.4,
            -0.25, -0.4, -2.0, -2.3, -0.6, 0.4,
            0, 0, 0, 0, 0, 0
        ])
    # 오른쪽 다리 (10~15)
    #[hip_x, hip_y, hip_z, knee, ankle_y, foot_y]

    # 왼쪽 다리 (16~21)
    #[hip_x, hip_y, hip_z, knee, ankle_y, foot_y]

    #view_single_squat_pose(xml_path, squat_qpos)
    situp_down_qpos = np.array([
        -0.4, 0, 0.09122,   # 좌표
        0.322788, 0, -0.19107, 0,                     # orientation
        0, -1.5, 0,                                   # 허리 회전, 상체 기울기, 상체 회전
        0.0182, 0.0142, 0.3, -1.9, -0.44, -0.02,      # 다리벌림, 다리 회전 각도, 골반 각도, 무릎 각도, 발목 회전, 발끝 회전 (왼쪽)
        0.0182, 0.0142, 0.3, -1.9, -0.44, -0.02,      # 다리벌림, 다리 회전 각도, 골반 각도, 무릎 각도, 발목 회전, 발끝 회전 (오른쪽)
        -0.1, 0.2, 0.7,                            #  , 팔 올림정도, 팔뚝 굽힙
        -0.1, 0.2, 0.7
    ])

    situp_up_qpos = np.array([
        -0.4, 0, 0.39122,   # 좌표
        0.322788, 0, 0.09107, 0,                     # orientation
        0, -1.5, 0,                                   # 허리 회전, 상체 기울기, 상체 회전
        0.0182, 0.0142, -1.5, -1.9, -0.44, -0.02,      # 다리벌림, 다리 회전 각도, 골반 각도, 무릎 각도, 발목 회전, 발끝 회전 (왼쪽)
        0.0182, 0.0142, -1.5, -1.9, -0.44, -0.02,      # 다리벌림, 다리 회전 각도, 골반 각도, 무릎 각도, 발목 회전, 발끝 회전 (오른쪽)
        -0.1, 0.2, 0.7,                            #  , 팔 올림정도, 팔뚝 굽힙
        -0.1, 0.2, 0.7
    ])

    #view_single_squat_pose(xml_path, situp_up_qpos)
    view_single_squat_pose(xml_path, standing_qpos)
    #view_single_squat_pose(xml_path, supine_hands_head_qpos)
