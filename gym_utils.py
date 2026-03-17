"""
gym_utils.py — Mario RL Platform
=================================
Compatible con:
  gym 0.26.2 + nes-py 8.2.1 + gym-super-mario-bros 7.4.0
  stable-baselines3 2.3.2 + gymnasium 0.29.1

DECISIÓN DE DISEÑO:
  NO usamos shimmy.GymV21CompatibilityV0 porque transpone el observation
  space de (n_stack,H,W) float32  →  (H,W,n_stack) int64, rompiendo
  la compatibilidad con modelos ya entrenados.

  En su lugar implementamos un wrapper gymnasium nativo mínimo
  (SB3GymnasiumWrapper) que satisface la interfaz que SB3 2.x necesita
  sin alterar el observation space.
"""

import numpy as np
import time

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

import gymnasium
from stable_baselines3.common.vec_env import DummyVecEnv

ACTION_SETS = {
    "SIMPLE":  SIMPLE_MOVEMENT,
    "RIGHT":   RIGHT_ONLY,
    "COMPLEX": COMPLEX_MOVEMENT,
}

_MAX_STEPS = 9999


def _make_base_env(env_id: str):
    """
    Crea env base y garantiza que NO haya TimeLimit de gym 0.26
    que devuelva 5-tuple incompatible con nes-py (4-tuple).

    Estrategia: crear el env y si tiene TimeLimit encima, quitarlo
    bajando al env interno.
    """
    try:
        env = gym.make(env_id, max_episode_steps=None)
    except TypeError:
        env = gym.make(env_id)

    # Si gym igual añadió TimeLimit, bajamos al env interno
    # TimeLimit está en gym.wrappers.time_limit.TimeLimit
    import gym.wrappers.time_limit as _tl
    while isinstance(env, _tl.TimeLimit):
        env = env.env

    return env


# ── Wrapper RAM → grid numérico ───────────────────────────────────────────────
class SMBRamWrapper(gym.Wrapper):
    """
    Convierte obs RGB → grid numérico (n_stack, H, W) float32.

    step()  → 4-tuple exacto  (obs, reward, done, info)
    reset() → numpy array     (NO tuple)
    """

    SCREEN_W = 16
    SCREEN_H = 13

    def __init__(self, env, crop_dim, n_stack=4, n_skip=4, max_steps=_MAX_STEPS):
        super().__init__(env)
        self.x0, self.x1, self.y0, self.y1 = crop_dim
        self.n_stack    = n_stack
        self.n_skip     = n_skip
        self.max_steps  = max_steps
        self._step_count = 0

        h = self.y1 - self.y0   # 13
        w = self.x1 - self.x0   # 16

        # shape = (n_stack, H, W) float32  ← NO cambiar, los modelos dependen de esto
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=2.0,
            shape=(n_stack, h, w),
            dtype=np.float32,
        )
        self.action_space = env.action_space
        self._frames = np.zeros((n_stack, h, w), dtype=np.float32)

    def _ram_to_grid(self):
        ram = self.env.unwrapped.ram
        mario_level_x = int(ram[0x6D]) * 256 + int(ram[0x86])
        mario_x       = int(ram[0x3AD])
        mario_y       = int(ram[0x3B8]) + 16
        x_start       = mario_level_x - mario_x

        grid = np.zeros((self.SCREEN_H, self.SCREEN_W), dtype=np.float32)
        screen_start = int(np.rint(x_start / 16))

        for i in range(self.SCREEN_W):
            for j in range(self.SCREEN_H):
                x_loc = (screen_start + i) % (self.SCREEN_W * 2)
                page  = x_loc // 16
                xl    = x_loc % 16
                addr  = 0x500 + xl + (page * 13 + j) * 16
                if ram[addr] != 0:
                    grid[j, i] = 1.0

        mx = (mario_x + 8) // 16
        my = (mario_y - 32) // 16
        if 0 <= mx < self.SCREEN_W and 0 <= my < self.SCREEN_H:
            grid[my, mx] = 2.0

        for k in range(5):
            if ram[0x0F + k] == 1:
                ex = int(ram[0x6E + k]) * 256 + int(ram[0x87 + k]) - x_start
                ey = int(ram[0xCF + k])
                ex_loc = (ex + 8) // 16
                ey_loc = (ey + 8 - 32) // 16
                if 0 <= ex_loc < self.SCREEN_W and 0 <= ey_loc < self.SCREEN_H:
                    grid[ey_loc, ex_loc] = -1.0

        return grid[self.y0:self.y1, self.x0:self.x1]

    def _push_frame(self, frame):
        self._frames = np.roll(self._frames, shift=-1, axis=0)
        self._frames[-1] = frame
        return self._frames.copy()

    def observation(self, obs):
        return self._ram_to_grid()

    def reset(self, **kwargs):
        self._step_count = 0
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            result = result[0]
        self._frames[:] = 0.0
        frame = self._ram_to_grid()
        for i in range(self.n_stack):
            self._frames[i] = frame
        return self._frames.copy()   # numpy array, NO tuple

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.n_skip):
            raw = self.env.step(action)
            if len(raw) == 5:
                _, reward, terminated, truncated, info = raw
                done = bool(terminated or truncated)
            else:
                _, reward, done, info = raw
                done = bool(done)
            total_reward += float(reward)
            if done:
                break
        self._step_count += self.n_skip
        if self._step_count >= self.max_steps:
            done = True
        self._push_frame(self._ram_to_grid())
        return self._frames.copy(), total_reward, done, info   # 4-tuple exacto


# ── Wrapper gymnasium mínimo para SB3 2.x ────────────────────────────────────
class SB3GymnasiumWrapper(gymnasium.Env):
    """
    Envuelve SMBRamWrapper en la interfaz gymnasium que SB3 2.x espera,
    SIN alterar el observation_space ni el dtype.

    A diferencia de shimmy.GymV21CompatibilityV0, este wrapper:
      - Preserva shape (n_stack, H, W) y dtype float32
      - reset() devuelve (obs, info) como pide gymnasium
      - step()  devuelve (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env: SMBRamWrapper):
        super().__init__()
        self._env = env

        # Copiar spaces a gymnasium preservando dtype y shape exactos
        obs_sp = env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=float(obs_sp.low.flat[0]),
            high=float(obs_sp.high.flat[0]),
            shape=obs_sp.shape,       # (n_stack, H, W)
            dtype=np.float32,         # siempre float32
        )
        act_sp = env.action_space
        self.action_space = gymnasium.spaces.Discrete(act_sp.n)

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), bool(done), False, info

    def render(self):
        return self._env.env.render(mode='rgb_array')

    def close(self):
        self._env.close()

    def get_raw_env(self):
        return self._env


# ── Factory principal ─────────────────────────────────────────────────────────
def load_smb_env(env_id: str, crop_dim, n_stack=4, n_skip=4,
                 action_set="SIMPLE", n_envs=1):
    """
    Construye entorno Mario → VecEnv para SB3 2.x.

    n_envs > 1 usa SubprocVecEnv (procesos paralelos) para aprovechar
    múltiples cores CPU y mantener la GPU ocupada.
    Recomendado: n_envs=8 para RTX 4060, n_envs=4 para CPU solo.

    observation_space: Box(-1, 2, (n_stack, H, W), float32)
    """
    actions = ACTION_SETS.get(action_set.upper(), SIMPLE_MOVEMENT)

    def _make():
        e = _make_base_env(env_id)
        e = JoypadSpace(e, actions)
        e = SMBRamWrapper(e, crop_dim, n_stack=n_stack, n_skip=n_skip)
        return SB3GymnasiumWrapper(e)

    # DummyVecEnv con n_envs instancias — más estable que SubprocVecEnv
    # con nes-py porque evita problemas de fork/pickle con el emulador.
    # El beneficio real de n_envs>1 es tener batches más grandes para la GPU,
    # lo cual se logra igual con DummyVecEnv.
    return DummyVecEnv([_make] * n_envs)


# ── Helper de alto nivel ──────────────────────────────────────────────────────
class SMB:
    def __init__(self, env, model):
        self.env   = env
        self.model = model

    def play(self, episodes=5, deterministic=True, render=False,
             return_eval=False):
        scores = []
        for ep in range(1, episodes + 1):
            states = self.env.reset()
            done   = False
            score  = 0.0
            while not done:
                if render:
                    self.env.render()
                action, _ = self.model.predict(states, deterministic=deterministic)
                states, reward, done, info = self.env.step(action)
                score += float(reward)
                if render:
                    time.sleep(0.01)
            scores.append(score)
            print(f"Episode {ep}: score={score:.1f}")
        if return_eval:
            return np.mean(scores), np.std(scores)
        return scores