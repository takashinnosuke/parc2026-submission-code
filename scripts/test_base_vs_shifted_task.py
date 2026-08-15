"""
デバッグ用スクリプト: T1公式タスク(table_27等のテクスチャ/照明シフト付き)と、
同じ物理タスクの「素の」(標準LIBEROベンチマークの無加工)バージョンとで
成功率を比較し、0%成功の主因が「視覚OOD(見た目の分布シフト)」か
「そもそもモデル/学習が弱い」かを切り分ける。

使い方 (harness venv内、Dockerコンテナ経由):
    python scripts/test_base_vs_shifted_task.py --server-url http://127.0.0.1:8000
"""
import argparse
import sys

import numpy as np


def find_base_task(language_substr: str):
    """標準LIBEROベンチマーク(libero_object等)から言語指示が一致するタスクを探す。"""
    from libero.libero import benchmark as libero_benchmark

    bd = libero_benchmark.get_benchmark_dict()
    for suite_name in ["libero_object", "libero_spatial", "libero_goal", "libero_10", "libero_90"]:
        if suite_name not in bd:
            continue
        suite = bd[suite_name](task_order_index=0)
        for i in range(suite.get_num_tasks()):
            task = suite.get_task(i)
            if language_substr.lower() in task.language.lower():
                return suite, i, task
    return None, None, None


def run_episode(client, env_manager, task_info, init_state, max_steps, label):
    from pipeline.rollout import RolloutExecutor
    from pipeline.config import EvalConfig, PerturbationConfig

    env = env_manager.create_env(task_info)
    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()

    action_dim = env.robots[0].action_dim
    obs, _, _, _ = env.step(np.zeros(action_dim))
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(action_dim))

    client.reset(instruction=task_info.language, seed=42)

    done = False
    for step in range(max_steps):
        action = client.get_action(obs)
        obs, reward, done, info = env.step(action)
        if step % 20 == 0:
            print(f"  [{label}] step {step}: action={np.array2string(action, precision=3)}")
        if done:
            break
    env.close()
    print(f"[{label}] Result: {'SUCCESS' if done else 'FAILURE'} (steps={step + 1})")
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--task-substr", default="tomato sauce")
    parser.add_argument("--t1-task-name", default="pick_up_the_tomato_sauce_and_place_it_in_the_basket_table_27")
    args = parser.parse_args()

    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager
    from pipeline.remote_policy import RemotePolicyClient

    config = EvalConfig(seed=42)
    env_manager = EnvironmentManager(config)
    client = RemotePolicyClient(server_url=args.server_url, timeout_sec=args.timeout)
    client.wait_for_server()

    results = {}

    # 1. T1公式タスク(テクスチャ/照明シフトあり)
    t1_tasks = env_manager.get_task_infos("libero_t1")
    t1_task = next(t for t in t1_tasks if t.name == args.t1_task_name)
    print(f"\n=== [SHIFTED] {t1_task.name} ===")
    print(f"Instruction: {t1_task.language}")
    init_states_shifted = env_manager.get_perturbed_init_states(
        t1_task, __import__("pipeline.config", fromlist=["PerturbationConfig"]).PerturbationConfig(), 1
    )
    results["shifted"] = run_episode(
        client, env_manager, t1_task, init_states_shifted[0], args.max_steps, "SHIFTED(table_27)"
    )

    # 2. 素の標準LIBEROタスク(テクスチャ変更なし)
    suite, idx, base_task = find_base_task(args.task_substr)
    if suite is None:
        print(f"\n[WARN] Could not find a base LIBERO task matching '{args.task_substr}'. Skipping base comparison.")
    else:
        from pipeline.environment import TaskInfo
        bddl_path = suite.get_task_bddl_file_path(idx)
        init_states_base = suite.get_task_init_states(idx)
        base_task_info = TaskInfo(
            task_id=idx, name=base_task.name, language=base_task.language,
            bddl_file=bddl_path, init_states=init_states_base, benchmark_name="base",
        )
        print(f"\n=== [BASE] {base_task.name} ===")
        print(f"Instruction: {base_task.language}")
        results["base"] = run_episode(
            client, env_manager, base_task_info, init_states_base[0], args.max_steps, "BASE(no shift)"
        )

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {'SUCCESS' if v else 'FAILURE'}")


if __name__ == "__main__":
    sys.exit(main())
