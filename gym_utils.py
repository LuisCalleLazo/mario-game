"""
gym_utils.py — Mario RL Platform
=================================
Compatible con:
  gym 0.26.2 + nes-py 8.2.1 + gym-super-mario-bros 7.4.0
  shimmy[gym-v21] 0.2.1
  stable-baselines3 2.3.2 + gymnasium 0.29.1

PROBLEMA RAÍZ:
  gym 0.26 TimeLimit.step() intenta desempacar 5 valores del env interno,
  pero nes-py devuelve 4. Solución: registrar el env con max_episode_steps=None
  para que gym NO añada TimeLimit, y manejar el timeout en SMBRamWrapper.
"""

import numpy as np
import time
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv

ACTION_SETS = {
    "SIMPLE":  SIMPLE_MOVEMENT,
    "RIGHT":   RIGHT_ONLY,
    "COMPLEX": COMPLEX_MOVEMENT,
}

# Timeout manual (equivale al max_episode_steps original de SMB = 9999 pasos)
_MAX_STEPS = 9999


def _make_base_env(env_id: str):
    """
    Crea el entorno base SIN TimeLimit wrapper de gym.

    gym.make() añade TimeLimit automáticamente; en gym 0.26 ese TimeLimit
    usa la API de 5 valores que es incompatible con nes-py (4 valores).
    Solución: override max_episode_steps=None para saltarse TimeLimit.
    """
    env = gym.make(env_id)

    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    return env


class SMBRamWrapper(gym.Wrapper):
    """
    Wrapper principal: convierte obs RGB -> grid RAM numérico.

    Garantías de API para shimmy.GymV21CompatibilityV0:
      reset() -> numpy.ndarray  (NO tuple)
      step()  -> (obs, reward, done, info)  (exactamente 4 valores)

    Maneja timeout manual en lugar de depender de gym TimeLimit.
    """

    SCREEN_W = 16
    SCREEN_H = 13

    def __init__(self, env, crop_dim, n_stack=4, n_skip=4, max_steps=_MAX_STEPS):
        super().__init__(env)
        self.x0, self.x1, self.y0, self.y1 = crop_dim
        self.n_stack   = n_stack
        self.n_skip    = n_skip
        self.max_steps = max_steps
        self._step_count = 0

        h = self.y1 - self.y0
        w = self.x1 - self.x0
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=2.0, shape=(n_stack, h, w), dtype=np.float32
        )
        self.action_space = env.action_space
        self._frames = np.zeros((n_stack, h, w), dtype=np.float32)

    # ── RAM -> grid ──────────────────────────────────────────────────────────
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

    # ── API gym (lo que shimmy llama directamente) ────────────────────────────
    def reset(self, **kwargs):
        """Devuelve SOLO numpy array. shimmy NO espera tuple."""
        self._step_count = 0
        # nes-py reset() devuelve solo obs (numpy array RGB)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            result = result[0]
        self._frames[:] = 0.0
        frame = self._ram_to_grid()
        for i in range(self.n_stack):
            self._frames[i] = frame
        return self._frames.copy()

    def step(self, action):
        """
        Devuelve EXACTAMENTE (obs, reward, done, info) — 4 valores.
        shimmy hace: obs, reward, done, info = self.gym_env.step(action)
        """
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.n_skip):
            # nes-py -> JoypadSpace siempre devuelve 4 valores
            # (gym TimeLimit eliminado con max_episode_steps=None)
            raw = self.env.step(action)

            if len(raw) == 4:
                _, reward, done, info = raw
                done = bool(done)
            elif len(raw) == 5:
                # Por si acaba llegando un TimeLimit de algún lado
                _, reward, terminated, truncated, info = raw
                done = bool(terminated or truncated)
            else:
                raise RuntimeError(f"env.step devolvió {len(raw)} valores")

            total_reward += float(reward)
            if done:
                break

        self._step_count += self.n_skip
        # Timeout manual (reemplaza gym TimeLimit)
        if self._step_count >= self.max_steps:
            done = True

        frame = self._ram_to_grid()
        self._push_frame(frame)

        return self._frames.copy(), total_reward, done, info   # 4-tuple exacto


# ── Factory ───────────────────────────────────────────────────────────────────
def load_smb_env(env_id: str, crop_dim, n_stack=4, n_skip=4,
                 action_set="SIMPLE"):
    """
    Construye entorno Mario -> DummyVecEnv para SB3 2.x.

    Stack:
      _make_base_env()          sin TimeLimit de gym
      JoypadSpace               reduce acciones
      SMBRamWrapper             RAM->grid, garantiza 4-tuple en step()
      GymV21CompatibilityV0     gym->gymnasium para SB3
      DummyVecEnv               vectorizado
    """
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
    actions = ACTION_SETS.get(action_set.upper(), SIMPLE_MOVEMENT)

    def _make():
        e = _make_base_env(env_id)          # SIN gym TimeLimit
        e = JoypadSpace(e, actions)
        e = SMBRamWrapper(e, crop_dim, n_stack=n_stack, n_skip=n_skip)
        return GymV21CompatibilityV0(env=e) # gym->gymnasium
    
    return DummyVecEnv([_make])


# ── Clase helper ──────────────────────────────────────────────────────────────
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