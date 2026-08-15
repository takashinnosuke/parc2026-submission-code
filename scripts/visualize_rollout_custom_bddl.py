"""visualize_rollout.py の変種。T1_TASKS.csv に登録されていない
LIBERO-plusのbddlファイル(テクスチャ摂動バリアント等)を直接指定して
1エピソード分のロールアウトを可視化する。

使い方:
    python scripts/visualize_rollout_custom_bddl.py --server-url http://127.0.0.1:8000 \
        --problem-folder libero_spatial \
        --bddl-name pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_table_16 \
        --instruction "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate" \
        --max-steps 150 --out /tmp/rollout.gif
"""
import argparse
import os
import re
import sys

import numpy as np

_SUFFIX_RE = re.compile(r"_light_[^.]*|_(?:table|tb)_\d+")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--problem-folder", required=True, help="libero_spatial / libero_object / libero_goal")
    parser.add_argument("--bddl-name", required=True, help="拡張子なしのbddlファイル名(=task_id)")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default="/tmp/rollout.gif")
    parser.add_argument("--out-actions", default="/tmp/rollout_actions.txt")
    args = parser.parse_args()

    import torch
    from libero.libero import benchmark as _b
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pipeline.remote_policy import RemotePolicyClient
    import imageio.v2 as imageio

    # compe/t1/register.py と同じロジック: perturbedシーンはベースと同じinit_statesを
    # 共有するため、テクスチャ/照明サフィックスを取り除いたファイル名でinit_statesを探す。
    stripped = _SUFFIX_RE.sub("", args.bddl_name)
    init_states_root = get_libero_path("init_states")
    init_states_path = os.path.join(init_states_root, args.problem_folder, f"{stripped}.pruned_init")
    init_states = torch.load(init_states_path, weights_only=False)

    bddl_root = get_libero_path("bddl_files")
    bddl_file_name = os.path.join(bddl_root, args.problem_folder, f"{args.bddl_name}.bddl")

    print(f"BDDL: {bddl_file_name}")
    print(f"Instruction: {args.instruction}")
    print(f"init_states (shared with base task): {init_states_path}")

    env_args = {
        "bddl_file_name": bddl_file_name,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)

    client = RemotePolicyClient(server_url=args.server_url, timeout_sec=args.timeout)
    client.wait_for_server()
    client.reset(instruction=args.instruction, seed=42)

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
            print(f"Episode done (SUCCESS, goal condition met) at step {step}")
            break

    env.close()

    if not done:
        print(f"Episode did not complete within {args.max_steps} steps (FAILURE)")

    if frames:
        imageio.mimsave(args.out, frames, fps=10)
        print(f"Saved {len(frames)} frames to {args.out}")
    with open(args.out_actions, "w") as f:
        f.write("\n".join(actions_log))
    print(f"Saved actions log to {args.out_actions}")


if __name__ == "__main__":
    sys.exit(main())
