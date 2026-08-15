"""visualize_rollout.py のRerun版。GIFではなく、カメラ映像・action各次元・state各次元・
done信号を時系列で同期表示できる.rrdファイルを出力する。

使い方:
    python scripts/visualize_rollout_rerun.py --server-url http://127.0.0.1:8000 \
        --task put_the_bowl_on_the_stove_light_11 \
        --max-steps 150 --out /tmp/rollout.rrd
"""
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default="/tmp/rollout.rrd")
    args = parser.parse_args()

    import rerun as rr
    from pipeline.config import EvalConfig, PerturbationConfig
    from pipeline.environment import EnvironmentManager
    from pipeline.remote_policy import RemotePolicyClient

    rr.init("parc2026_rollout", spawn=False)
    rr.save(args.out)

    config = EvalConfig(seed=42)
    env_manager = EnvironmentManager(config)
    task_infos = env_manager.get_task_infos("libero_t1")
    task_info = next(t for t in task_infos if t.name == args.task)
    print(f"Task: {task_info.name}")
    print(f"Instruction: {task_info.language}")

    env = env_manager.create_env(task_info)
    init_states = env_manager.get_perturbed_init_states(task_info, PerturbationConfig(), 1)

    client = RemotePolicyClient(server_url=args.server_url, timeout_sec=args.timeout)
    client.wait_for_server()
    client.reset(instruction=task_info.language, seed=42)

    env.reset()
    env.sim.set_state_from_flattened(init_states[0])
    env.sim.forward()

    action_dim = env.robots[0].action_dim
    obs, _, _, _ = env.step(np.zeros(action_dim))
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(action_dim))

    action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    for step in range(args.max_steps):
        rr.set_time("step", sequence=step)

        agent_img = obs.get("agentview_image")
        wrist_img = obs.get("robot0_eye_in_hand_image")
        if agent_img is not None:
            rr.log("camera/agentview", rr.Image(np.flipud(np.asarray(agent_img))))
        if wrist_img is not None:
            rr.log("camera/wrist", rr.Image(np.flipud(np.asarray(wrist_img))))

        gripper_qpos = obs.get("robot0_gripper_qpos")
        if gripper_qpos is not None:
            rr.log("state/gripper_qpos_finger0", rr.Scalars(float(np.asarray(gripper_qpos).reshape(-1)[0])))

        eef_pos = obs.get("robot0_eef_pos")
        if eef_pos is not None:
            p = np.asarray(eef_pos).reshape(-1)
            rr.log("state/eef_x", rr.Scalars(float(p[0])))
            rr.log("state/eef_y", rr.Scalars(float(p[1])))
            rr.log("state/eef_z", rr.Scalars(float(p[2])))

        action = client.get_action(obs)
        for name, val in zip(action_names, action):
            rr.log(f"action/{name}", rr.Scalars(float(val)))

        obs, reward, done, info = env.step(action)
        rr.log("episode/reward", rr.Scalars(float(reward)))
        rr.log("episode/done", rr.Scalars(1.0 if done else 0.0))

        if step % 20 == 0:
            print(f"step {step}: action={action}, reward={reward}, done={done}")
        if done:
            print(f"Episode done (SUCCESS) at step {step}")
            break

    env.close()
    print(f"Saved Rerun recording to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
