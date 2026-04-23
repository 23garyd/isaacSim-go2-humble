import os
import time
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

base_vel_cmd_input = None
_last_press_time = {}   # env_idx -> time.monotonic() of last PRESS; used to decay command after keys released
_STICKY_TIMEOUT = 0.2

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input, _last_press_time
    # Decay any env whose last PRESS is older than _STICKY_TIMEOUT.
    # Needed because some remote-desktop stacks expand a key-hold into rapid
    # PRESS+RELEASE pairs; relying on RELEASE to zero the command would race the
    # next PRESS within the same sim step. Instead, PRESS refreshes the timer.
    now = time.monotonic()
    for i in list(_last_press_time.keys()):
        if now - _last_press_time[i] > _STICKY_TIMEOUT:
            base_vel_cmd_input[i].zero_()
            del _last_press_time[i]
    return base_vel_cmd_input.clone().to(env.device)

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    global base_vel_cmd_input, _last_press_time
    lin_vel = 1.5
    ang_vel = 1.5

    if base_vel_cmd_input is None:
        return True
    # Only react to KEY_PRESS. Remote-desktop autorepeat emits a PRESS on every
    # repeat cycle, which is exactly what keeps _last_press_time fresh while held.
    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return True

    name = event.input.name
    now = time.monotonic()

    # env 0
    if name == 'W':
        base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
        _last_press_time[0] = now
    elif name == 'S':
        base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
        _last_press_time[0] = now
    elif name == 'A':
        base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
        _last_press_time[0] = now
    elif name == 'D':
        base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
        _last_press_time[0] = now
    elif name == 'Z':
        base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
        _last_press_time[0] = now
    elif name == 'C':
        base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        _last_press_time[0] = now

    # env 1 (only if >1 envs)
    if base_vel_cmd_input.shape[0] > 1:
        if name == 'I':
            base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
            _last_press_time[1] = now
        elif name == 'K':
            base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
            _last_press_time[1] = now
        elif name == 'J':
            base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
            _last_press_time[1] = now
        elif name == 'L':
            base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
            _last_press_time[1] = now
        elif name == 'M':
            base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
            _last_press_time[1] = now
        elif name == '>':
            base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
            _last_press_time[1] = now

    return True

def get_rsl_flat_policy(cfg):
    cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy

def get_rsl_rough_policy(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy
