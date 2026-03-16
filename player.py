"""
player.py — Mario RL Platform
==============================
Corre episodios de evaluación en background, captura cada frame
como imagen JPEG y lo emite por WebSocket al navegador.

El emulador no tiene pantalla (headless) así que usamos
env.render(mode='rgb_array') para obtener el numpy array del frame
y lo comprimimos a JPEG base64 para enviarlo por socket.
"""

import threading
import base64
import time
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from PIL import Image
    import io as _io
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _frame_to_b64(frame_rgb: np.ndarray, quality: int = 60) -> str:
    """Convierte numpy RGB array -> JPEG base64 string."""
    if _HAS_CV2:
        # BGR para cv2
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Escalar x2 para que se vea más grande en el browser
        bgr = cv2.resize(bgr, (512, 480), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode('utf-8')
    elif _HAS_PIL:
        img = Image.fromarray(frame_rgb)
        img = img.resize((512, 480), Image.NEAREST)
        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        raise RuntimeError("Instala opencv-python-headless o Pillow para streaming de frames")


class MarioPlayer:
    """
    Corre N episodios del modelo, emitiendo frames por WebSocket.

    emit_fn(event, data) — función de emisión de socketio
    """

    def __init__(self, model_path: str, world: int, level: int,
                 action_set: str, emit_fn, job_id: str,
                 episodes: int = 3, fps_cap: int = 30):
        self.model_path = model_path
        self.world      = world
        self.level      = level
        self.action_set = action_set
        self.emit_fn    = emit_fn
        self.job_id     = job_id
        self.episodes   = episodes
        self.fps_cap    = fps_cap
        self._stop      = threading.Event()
        self._thread    = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _emit(self, event: str, data: dict):
        try:
            self.emit_fn(event, {**data, "play_id": self.job_id})
        except Exception:
            pass

    def _run(self):
        from gym_utils import load_smb_env
        from stable_baselines3 import PPO
        import torch

        try:
            self._emit("play_status", {"status": "loading", "message": "Cargando modelo…"})

            device = "cuda" if torch.cuda.is_available() else "cpu"
            env    = load_smb_env(
                f"SuperMarioBros-{self.world}-{self.level}-v0",
                [0, 16, 0, 13],
                action_set=self.action_set,
            )
            model = PPO.load(self.model_path, env=env, device=device)

            frame_delay = 1.0 / self.fps_cap
            scores = []

            for ep in range(1, self.episodes + 1):
                if self._stop.is_set():
                    break

                self._emit("play_status", {
                    "status":  "playing",
                    "episode": ep,
                    "total":   self.episodes,
                    "message": f"Episodio {ep}/{self.episodes}",
                })

                obs     = env.reset()
                done    = False
                score   = 0.0
                step    = 0
                t_last  = time.time()

                while not done and not self._stop.is_set():
                    # Obtener frame RGB del emulador
                    # DummyVecEnv envuelve el env; accedemos al env interno
                    try:
                        frame_rgb = env.envs[0].env.gym_env.env.render(mode='rgb_array')
                    except Exception:
                        try:
                            frame_rgb = env.envs[0].env.gym_env.render(mode='rgb_array')
                        except Exception:
                            frame_rgb = None

                    if frame_rgb is not None:
                        try:
                            b64 = _frame_to_b64(frame_rgb)
                            self._emit("play_frame", {
                                "frame":   b64,
                                "episode": ep,
                                "step":    step,
                                "score":   round(score, 1),
                            })
                        except Exception:
                            pass

                    # Predecir y avanzar
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    score += float(reward)
                    step  += 1

                    # Cap de FPS para no saturar el socket
                    elapsed = time.time() - t_last
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                    t_last = time.time()

                scores.append(score)
                self._emit("play_status", {
                    "status":  "episode_done",
                    "episode": ep,
                    "score":   round(score, 1),
                    "message": f"Episodio {ep} terminado — score: {score:.0f}",
                })
                time.sleep(0.5)

            env.close()
            import numpy as _np
            self._emit("play_status", {
                "status":     "completed",
                "mean_score": round(float(_np.mean(scores)), 1),
                "std_score":  round(float(_np.std(scores)), 1),
                "scores":     [round(s, 1) for s in scores],
                "message":    "Evaluación completada",
            })

        except Exception as e:
            self._emit("play_status", {
                "status":  "error",
                "message": str(e),
            })
            raise