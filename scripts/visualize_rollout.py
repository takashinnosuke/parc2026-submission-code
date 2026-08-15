"""
1エピソード分のロールアウトを実行し、agentview / eye_in_hand 両カメラの
フレームをGIFに保存する。policy_server.py の実際の出力を目視確認するための
デバッグ専用スクリプト。

使い方 (harness venv / Dockerコンテナ内、pipeline import可能な状態で):
    python scripts/visualize_rollout.py --server-url http://127.0.0.1:8000 \
        --task pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_table_2 \
        --max-steps 150 --out /tmp/rollout.gif
"""
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default="/tmp/rollout.gif")
    parser.add_argument("--out-actions", default="/tmp/rollout_actions.txt")
    args = parser.parse_args()

    from pipeline.config import EvalConfig, PerturbationConfig
    from pipeline.environment import EnvironmentManager
    from pipeline.remote_policy import RemotePolicyClient
    import imageio.v2 as imageio

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

    frames = []
    actions_log = []
    for step in range(args.max_steps):
        agent_img = obs.get("agentview_image")
        wrist_img = obs.get("robot0_eye_in_hand_image")
        if agent_img is not None and wrist_img is not None:
            a = np.asarray(agent_img)
            w = np.asarray(wrist_img)
            combo = np.concatenate([a, w], axis=1)
            frames.append(combo.astype(np.uint8))

        action = client.get_action(obs)
        actions_log.append(np.array2string(action, precision=3))
        obs, reward, done, info = env.step(action)

        if step % 20 == 0:
            print(f"step {step}: action={action}, reward={reward}, done={done}")
        if done:
            print(f"Episode done at step {step}")
            break

    env.close()

    if frames:
        imageio.mimsave(args.out, frames, fps=10)
        print(f"Saved {len(frames)} frames to {args.out}")
    with open(args.out_actions, "w") as f:
        f.write("\n".join(actions_log))
    print(f"Saved actions log to {args.out_actions}")


if __name__ == "__main__":
    sys.exit(main())
