"""
デバッグ用: lerobot/libero_plusから抽出したGT action列をopen-loopでシミュレータに
再生し、把持自体が成功するか(=action空間・環境のデコードが正しいか)を確認する。
policyは一切使わない。

使い方 (harness venv内、Dockerコンテナ経由):
    python scripts/replay_gt_actions.py --actions-npy /out/gt_actions_tomato_ep28.npy \
        --out /out/gt_replay.gif
"""
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions-npy", required=True)
    parser.add_argument("--task-substr", default="tomato sauce")
    parser.add_argument("--out", default="/tmp/gt_replay.gif")
    parser.add_argument("--out-actions", default="/tmp/gt_replay_actions.txt")
    args = parser.parse_args()

    import imageio.v2 as imageio
    from libero.libero import benchmark as libero_benchmark
    from pipeline.environment import EnvironmentManager, TaskInfo
    from pipeline.config import EvalConfig

    actions = np.load(args.actions_npy)
    print(f"Loaded {actions.shape[0]} GT actions, dim={actions.shape[1]}")
    print(f"Gripper (last dim) trace: {np.array2string(actions[:, -1], precision=2)}")

    bd = libero_benchmark.get_benchmark_dict()
    suite, idx, task = None, None, None
    for suite_name in ["libero_object", "libero_spatial", "libero_goal", "libero_10", "libero_90"]:
        if suite_name not in bd:
            continue
        s = bd[suite_name](task_order_index=0)
        for i in range(s.get_num_tasks()):
            t = s.get_task(i)
            if args.task_substr.lower() in t.language.lower():
                suite, idx, task = s, i, t
                break
        if suite is not None:
            break

    if suite is None:
        print(f"[ERROR] Could not find a task matching '{args.task_substr}'")
        sys.exit(1)

    print(f"Using task: {task.name} | {task.language}")

    config = EvalConfig(seed=42)
    env_manager = EnvironmentManager(config)
    bddl_path = suite.get_task_bddl_file_path(idx)
    init_states = suite.get_task_init_states(idx)
    task_info = TaskInfo(
        task_id=idx, name=task.name, language=task.language,
        bddl_file=bddl_path, init_states=init_states, benchmark_name="base",
    )

    env = env_manager.create_env(task_info)
    env.reset()
    env.sim.set_state_from_flattened(init_states[0])
    env.sim.forward()

    action_dim = env.robots[0].action_dim
    obs, _, _, _ = env.step(np.zeros(action_dim))
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(action_dim))

    frames = []
    actions_log = []
    done = False
    for step, action in enumerate(actions):
        a = np.asarray(action, dtype=np.float32)
        if a.shape[0] < action_dim:
            a = np.pad(a, (0, action_dim - a.shape[0]))
        else:
            a = a[:action_dim]

        agent_img = obs.get("agentview_image")
        wrist_img = obs.get("robot0_eye_in_hand_image")
        if agent_img is not None and wrist_img is not None:
            combo = np.concatenate([np.asarray(agent_img), np.asarray(wrist_img)], axis=1)
            frames.append(combo.astype(np.uint8))

        obs, reward, done, info = env.step(a)
        actions_log.append(np.array2string(a, precision=3))
        if step % 20 == 0:
            print(f"  step {step}: action={a}, reward={reward}, done={done}")
        if done:
            print(f"GT replay reached success (done=True) at step {step}")
            break

    env.close()
    print(f"\n=== RESULT: {'SUCCESS' if done else 'FAILURE'} (steps={step + 1}/{len(actions)}) ===")

    if frames:
        imageio.mimsave(args.out, frames, fps=10)
        print(f"Saved {len(frames)} frames to {args.out}")
    with open(args.out_actions, "w") as f:
        f.write("\n".join(actions_log))


if __name__ == "__main__":
    sys.exit(main())
