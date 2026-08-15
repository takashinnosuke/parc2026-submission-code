"""
デバッグ用: ポリシーのロールアウト時のeef軌道を、同一タスク・同一初期状態での
GT(デモ)軌道と比較し、どの時点からどの程度ズレ始めるかを定量化する。

使い方 (harness venv内、Dockerコンテナ経由):
    python scripts/compare_rollout_trajectory.py --server-url http://127.0.0.1:8000 \
        --gt-states-npy /out/gt_states_tomato_ep28.npy \
        --task-substr "tomato sauce" --max-steps 148
"""
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--gt-states-npy", required=True)
    parser.add_argument("--task-substr", default="tomato sauce")
    parser.add_argument("--max-steps", type=int, default=148)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out-csv", default="/tmp/trajectory_compare.csv")
    args = parser.parse_args()

    from libero.libero import benchmark as libero_benchmark
    from pipeline.environment import EnvironmentManager, TaskInfo
    from pipeline.config import EvalConfig
    from pipeline.remote_policy import RemotePolicyClient

    gt_states = np.load(args.gt_states_npy)
    gt_eef_pos = gt_states[:, :3]
    gt_gripper = gt_states[:, -1] if gt_states.shape[1] >= 8 else None
    print(f"Loaded GT states: {gt_states.shape}")

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
        print(f"[ERROR] no task matching '{args.task_substr}'")
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

    client = RemotePolicyClient(server_url=args.server_url, timeout_sec=args.timeout)
    client.wait_for_server()
    client.reset(instruction=task_info.language, seed=42)

    policy_eef_pos = []
    policy_gripper = []
    done = False
    for step in range(args.max_steps):
        eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)))
        policy_eef_pos.append(eef_pos.copy())

        action = client.get_action(obs)
        policy_gripper.append(float(action[-1]))
        obs, reward, done, info = env.step(action)

        if step % 20 == 0:
            print(f"  step {step}: eef_pos={eef_pos}, gripper_action={action[-1]:.3f}")
        if done:
            print(f"Policy rollout SUCCESS at step {step}")
            break
    env.close()

    policy_eef_pos = np.array(policy_eef_pos)
    n = min(len(policy_eef_pos), len(gt_eef_pos))

    print(f"\n=== Trajectory divergence (policy vs GT, first {n} steps) ===")
    print(f"{'step':>5} {'||policy-gt|| (m)':>18} {'policy_gripper':>15} {'gt_gripper':>12}")
    dists = []
    with open(args.out_csv, "w") as f:
        f.write("step,dist,policy_gripper,gt_gripper\n")
        for i in range(n):
            d = float(np.linalg.norm(policy_eef_pos[i] - gt_eef_pos[i]))
            dists.append(d)
            pg = policy_gripper[i] if i < len(policy_gripper) else float("nan")
            gg = gt_gripper[i] if gt_gripper is not None else float("nan")
            f.write(f"{i},{d:.4f},{pg:.3f},{gg:.3f}\n")
            if i % 10 == 0 or i == n - 1:
                print(f"{i:5d} {d:18.4f} {pg:15.3f} {gg:12.3f}")

    dists = np.array(dists)
    print(f"\nMean distance: {dists.mean():.4f} m")
    print(f"Step0 distance: {dists[0]:.4f} m (initial state alignment check)")
    print(f"Distance at step 10: {dists[min(10, n-1)]:.4f} m")
    print(f"Distance at step 30: {dists[min(30, n-1)]:.4f} m")
    print(f"Distance at step 60: {dists[min(60, n-1)]:.4f} m")
    print(f"Max distance: {dists.max():.4f} m at step {int(dists.argmax())}")
    print(f"Result: {'SUCCESS' if done else 'FAILURE'}")
    print(f"Saved CSV to {args.out_csv}")


if __name__ == "__main__":
    sys.exit(main())
