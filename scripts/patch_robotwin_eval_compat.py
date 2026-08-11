#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one `{label}` block, found {count}. "
            "The RoboTwin file may use an unsupported revision."
        )
    return text.replace(old, new, 1)


def patch_eval_policy(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if (
        "_result_suffix_from_task_config" in text
        and 'usr_args.get("eval_output_dir")' in text
        and 'usr_args.get("eval_num_episodes"' in text
    ):
        print(f"Already patched: {path}")
        return

    text = replace_once(
        text,
        """def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
""",
        """def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(f"Unsupported task_config for evaluation: {task_config}")


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
""",
        "main header",
    )
    text = replace_once(
        text,
        """    instruction_type = usr_args["instruction_type"]
    save_dir = None
""",
        """    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(
        usr_args.get("skip_get_obs_within_replan", False)
    )
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(
            f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}"
        )
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
""",
        "runtime evaluation arguments",
    )
    text = replace_once(
        text,
        """    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
""",
        """    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if "eval_video_log" in usr_args:
        args["eval_video_log"] = parse_bool(usr_args["eval_video_log"])

    args['task_name'] = task_name
""",
        "eval video override",
    )
    text = replace_once(
        text,
        """    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)
""",
        """    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(
            f"eval_result/{task_name}/{policy_name}/{task_config}/"
            f"{ckpt_setting}/{eval_ts}"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
""",
        "evaluation output directory",
    )
    text = replace_once(
        text,
        "    test_num = 100\n",
        "    test_num = eval_num_episodes\n",
        "episode count",
    )
    text = replace_once(
        text,
        """                                   video_size=video_size,
                                   instruction_type=instruction_type)
""",
        """                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan)
""",
        "eval_policy call",
    )
    text = replace_once(
        text,
        """    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\\n\\n")
""",
        """    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\\n\\n")
""",
        "result file",
    )
    text = replace_once(
        text,
        """                video_size=None,
                instruction_type=None):
""",
        """                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False):
""",
        "eval_policy signature",
    )
    text = replace_once(
        text,
        """        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
""",
        """        args["render_freq"] = render_freq

        try:
            TASK_ENV.setup_demo(
                now_ep_num=now_id, seed=now_seed, is_test=True, **args
            )
        except UnStableError:
            succ_seed -= 1
            if suc_test_seed_list and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            TASK_ENV.close_env()
            now_seed += 1
            continue
        except Exception:
            succ_seed -= 1
            if suc_test_seed_list and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            print("Evaluation setup error:", traceback.format_exc())
            TASK_ENV.close_env()
            now_seed += 1
            continue
        episode_info_list = [episode_info["info"]]
""",
        "evaluation setup retry",
    )
    text = replace_once(
        text,
        """        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
""",
        """        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(
                model, "should_request_observation"
            ):
                need_obs = bool(model.should_request_observation())
            observation = TASK_ENV.get_obs() if need_obs else None
            eval_func(TASK_ENV, model, observation)
""",
        "observation reuse",
    )

    path.write_text(text, encoding="utf-8")
    print(f"Patched: {path}")


def patch_camera_override(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "FASTWAM_CAMERA_TYPE_OVERRIDE"
    if marker in text:
        print(f"Camera override already patched: {path}")
        return
    text = replace_once(
        text,
        """    if "eval_video_log" in usr_args:
        args["eval_video_log"] = parse_bool(usr_args["eval_video_log"])

    args['task_name'] = task_name
""",
        """    if "eval_video_log" in usr_args:
        args["eval_video_log"] = parse_bool(usr_args["eval_video_log"])

    # FASTWAM_CAMERA_TYPE_OVERRIDE: match the resolution used by training data.
    camera_type = usr_args.get("camera_type")
    if camera_type is not None and str(camera_type).strip() != "":
        args["camera"]["head_camera_type"] = str(camera_type)
        args["camera"]["wrist_camera_type"] = str(camera_type)

    args['task_name'] = task_name
""",
        "camera type override",
    )
    path.write_text(text, encoding="utf-8")
    print(f"Patched camera override: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch an upstream RoboTwin checkout for FastWAM evaluation."
    )
    parser.add_argument(
        "--robotwin-root",
        default="checkpoints/RoboTwin",
        help="RoboTwin repository root.",
    )
    args = parser.parse_args()
    root = Path(args.robotwin_root).expanduser().resolve()
    target = root / "script" / "eval_policy.py"
    if not target.is_file():
        raise FileNotFoundError(f"RoboTwin eval policy not found: {target}")
    patch_eval_policy(target)
    patch_camera_override(target)


if __name__ == "__main__":
    main()
