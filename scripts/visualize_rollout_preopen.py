"""visualize_rollout.py の A/B 版。

harness の warmup (env.step(zeros)) は gripper action=0 を送るため、Pandaグリッパが
「全開 qpos=0.0392」から「半開 qpos=0.0200」まで閉じてしまう。デモ(学習データ)の
エピソード先頭は必ず全開(≈0.039)なので、ポリシーは初手から未学習の状態を見ることになる。

--pre-open N を指定すると、ポリシーに制御を渡す前に「移動量ゼロ + gripper=-1(open)」を
N ステップ送り、グリッパを全開に戻してから rollout を開始する。
"""
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--pre-open", type=int, default=25)
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
    for _ in range(11):
        obs, _, _, _ = env.step(np.zeros(action_dim))
    print(f"after zero-warmup : gripper_qpos={np.round(obs['robot0_gripper_qpos'], 5)}")

    if args.pre_open > 0:
        open_action = np.zeros(action_dim)
        open_action[-1] = -1.0
        for _ in range(args.pre_open):
            obs, _, _, _ = env.step(open_action)
        print(f"after pre-open x{args.pre_open}: gripper_qpos={np.round(obs['robot0_gripper_qpos'], 5)}")

    frames = []
    actions_log = []
    for step in range(args.max_steps):
        a_img = obs.get("agentview_image")
        w_img = obs.get("robot0_eye_in_hand_image")
        if a_img is not None and w_img is not None:
            frames.append(np.concatenate([np.asarray(a_img), np.asarray(w_img)], axis=1).astype(np.uint8))

        action = client.get_action(obs)
        actions_log.append(
            f"{np.array2string(action, precision=3)}  gq={obs['robot0_gripper_qpos'][0]:.4f}"
        )
        obs, reward, done, info = env.step(action)

        if step % 10 == 0:
            print(f"step {step}: gripper_act={action[-1]:+.3f} gq={obs['robot0_gripper_qpos'][0]:.4f} "
                  f"eef={np.round(obs['robot0_eef_pos'], 3)} reward={reward} done={done}")
        if done:
            print(f"Episode SUCCESS/done at step {step}")
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
