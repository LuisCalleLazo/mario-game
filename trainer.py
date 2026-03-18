"""
trainer.py
==========
Motor de entrenamiento PPO para Super Mario Bros.

Características:
  - Checkpoints automáticos cada N pasos
  - Reanudación desde checkpoint o modelo final
  - Callback WebSocket para progreso en tiempo real
  - Persistencia de jobs en JSON (data/jobs.json)
  - Auto-detección de dispositivo: CUDA local / CPU en servidor
"""

import os
import time
import json
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gym_utils import load_smb_env

# ── Directorios base ──────────────────────────────────────────────────────────
# Si DATA_DIR apunta a /data (Docker) pero no existe o no hay permisos, usar ./data
_raw_data_dir = os.environ.get("DATA_DIR", "./data")
if _raw_data_dir.startswith("/") and not os.access(_raw_data_dir, os.W_OK):
    import warnings
    warnings.warn(f"DATA_DIR={_raw_data_dir} no tiene permisos, usando ./data")
    _raw_data_dir = "./data"
BASE_DIR        = Path(_raw_data_dir)
MODELS_DIR      = BASE_DIR / "models"
LOGS_DIR        = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
JOBS_FILE       = BASE_DIR / "jobs.json"

for d in (MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Persistencia JSON ─────────────────────────────────────────────────────────
_jobs_lock = threading.Lock()

def load_jobs_from_disk() -> dict:
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_jobs_to_disk(jobs: dict):
    with _jobs_lock:
        JOBS_FILE.write_text(
            json.dumps(jobs, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


# ── Callback de progreso ──────────────────────────────────────────────────────
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, emit_fn=None,
                 update_freq: int = 500, job_id: str = "",
                 initial_timesteps: int = 0):
        super().__init__(verbose=0)
        self.total_timesteps    = int(total_timesteps)   # pasos de ESTA sesión
        self.initial_timesteps  = int(initial_timesteps) # pasos previos del modelo
        self.emit_fn    = emit_fn or (lambda d: None)
        self.update_freq = update_freq
        self.job_id     = job_id
        self.start_time = None
        self.last_mean_reward = 0.0
        self._step_start = 0  # num_timesteps al inicio de esta sesión

    def _on_training_start(self):
        self.start_time  = time.time()
        # Guardar el num_timesteps actual para calcular progreso relativo
        self._step_start = self.num_timesteps
        self._publish({"status": "training", "timestep": self.initial_timesteps, "progress": 0.0})

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq == 0:
            # Pasos hechos EN ESTA SESIÓN (relativo)
            session_steps = self.num_timesteps - self._step_start
            progress = min(session_steps / max(self.total_timesteps, 1), 1.0)

            elapsed  = time.time() - self.start_time
            eta_sec  = (elapsed / max(progress, 1e-9)) * (1 - progress)

            if len(self.model.ep_info_buffer) > 0:
                self.last_mean_reward = float(
                    np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                )

            self._publish({
                "status":           "training",
                # timestep acumulado total (prev + actual)
                "timestep":         self.initial_timesteps + session_steps,
                "total":            self.initial_timesteps + self.total_timesteps,
                "session_timestep": session_steps,
                "session_total":    self.total_timesteps,
                "progress":         round(progress, 4),
                "elapsed_sec":      round(elapsed, 1),
                "eta_sec":          round(eta_sec, 1),
                "mean_reward":      round(self.last_mean_reward, 2),
            })
        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        session_steps = self.num_timesteps - self._step_start
        self._publish({
            "status":           "saving",
            "timestep":         self.initial_timesteps + session_steps,
            "total":            self.initial_timesteps + self.total_timesteps,
            "session_timestep": session_steps,
            "session_total":    self.total_timesteps,
            "progress":         1.0,
            "elapsed_sec":      round(elapsed, 1),
            "eta_sec":          0,
            "mean_reward":      round(self.last_mean_reward, 2),
        })

    def _publish(self, data: dict):
        data["job_id"] = self.job_id
        try:
            self.emit_fn(data)
        except Exception:
            pass


# ── Motor principal ───────────────────────────────────────────────────────────
class MarioTrainer:
    """
    Entrena PPO en background para un job dado.

    Parámetros de config (dict):
        world, level, action_set
        total_timesteps, learning_rate, n_steps, batch_size, n_epochs, gamma
        n_stack, n_skip
        checkpoint_freq
        resume_from      (ruta .zip opcional)
        device_override  ("cpu" | "cuda" | "auto")
        job_id
    """

    CROP_DIM = [0, 16, 0, 13]

    def __init__(self, config: dict, emit_fn=None, jobs_registry: dict = None):
        self.config        = config
        self.emit_fn       = emit_fn or (lambda d: None)
        self.jobs_registry = jobs_registry  # referencia al dict global de jobs
        self.job_id        = config.get("job_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._stop         = threading.Event()
        self._thread       = None
        self.error         = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def force_checkpoint(self):
        """
        Fuerza un checkpoint en el próximo step del entrenamiento
        escribiendo un archivo de señal que el callback detecta.
        NO requiere reiniciar Flask ni el trainer.
        """
        try:
            if not self.is_alive():
                return False, "Entrenamiento no activo"
            signals_dir = BASE_DIR / "signals"
            signals_dir.mkdir(parents=True, exist_ok=True)
            signal_file = signals_dir / f"{self.job_id}.ckpt"
            signal_file.touch()
            msg = "Señal de checkpoint enviada — se guardará en el próximo step"
            print(f"[Force Checkpoint] {msg}")
            self._emit_and_persist({
                "job_id":  self.job_id,
                "status":  "training",
                "message": msg,
            })
            return True, str(CHECKPOINTS_DIR / self.job_id)
        except Exception as e:
            return False, str(e)

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    # ── Detección de dispositivo ──────────────────────────────────────────────
    def _select_device(self) -> str:
        override = self.config.get("device_override", "auto").lower()
        if override in ("cpu", "cuda"):
            return override
        # auto: usar CUDA si está disponible y funciona
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()  # test rápido
                return "cuda"
            except Exception:
                pass
        return "cpu"

    def _build_env(self):
        c = self.config
        env_id  = f"SuperMarioBros-{c.get('world',1)}-{c.get('level',1)}-v0"
        n_envs  = int(c.get("n_envs", 1))
        device  = c.get("_device", "cpu")
        # En CPU limitar a 4 envs para no saturar cores
        if device == "cpu" and n_envs > 4:
            n_envs = 4
        return load_smb_env(
            env_id,
            self.CROP_DIM,
            n_stack=int(c.get("n_stack", 4)),
            n_skip=int(c.get("n_skip", 4)),
            action_set=c.get("action_set", "SIMPLE"),
            n_envs=n_envs,
        )

    def _build_model(self, env, device: str):
        c = self.config
        resume_path = c.get("resume_from", "").strip()
        if resume_path and os.path.exists(resume_path):
            print(f"[Trainer] Reanudando desde {resume_path} en {device}")
            # custom_objects evita el error de deserialización de clip_range/lr_schedule
            # cuando el modelo fue guardado con diferente versión de Python
            custom_objects = {
                "clip_range": 0.2,
                "lr_schedule": float(c.get("learning_rate", 3e-4)),
            }
            model = PPO.load(
                resume_path, env=env, device=device,
                custom_objects=custom_objects,
            )
            # Asignar tensorboard_log manualmente (no se guarda en el zip)
            log_dir = str(LOGS_DIR / self.job_id)
            os.makedirs(log_dir, exist_ok=True)
            model.tensorboard_log = log_dir
            return model

        n_steps   = int(c.get("n_steps", 2048))
        n_envs    = int(c.get("n_envs", 1))
        batch_size = int(c.get("batch_size", 64))
        # batch_size debe dividir n_steps * n_envs exactamente
        total_steps = n_steps * n_envs
        if total_steps % batch_size != 0:
            # Ajustar batch_size al divisor más cercano
            import math
            batch_size = max(64, total_steps // (total_steps // batch_size))
            print(f"[Trainer] batch_size ajustado a {batch_size} (n_steps*n_envs={total_steps})")

        return PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=0,
            learning_rate=float(c.get("learning_rate", 3e-4)),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(c.get("n_epochs", 10)),
            gamma=float(c.get("gamma", 0.99)),
            tensorboard_log=str(LOGS_DIR / self.job_id),
        )

    # ── Loop principal ────────────────────────────────────────────────────────
    def _run(self):
        try:
            device = self._select_device()
            self._update_job({"status": "building", "device": device,
                              "message": f"Construyendo entorno… (device={device})"})

            self.config["_device"] = device  # para que _build_env sepa el device
            env   = self._build_env()
            model = self._build_model(env, device)
            self._current_model = model  # referencia para force_checkpoint()

            total     = int(self.config.get("total_timesteps", 100_000))
            ckpt_freq = int(self.config.get("checkpoint_freq", 10_000))
            ckpt_dir  = CHECKPOINTS_DIR / self.job_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Detectar cuántos pasos tiene ya el modelo (si es reanudación)
            initial_ts = 0
            if hasattr(model, 'num_timesteps') and model.num_timesteps > 0:
                initial_ts = int(model.num_timesteps)
                print(f"[Trainer] Reanudando desde timestep {initial_ts:,}")

            progress_cb = ProgressCallback(
                total_timesteps=total,
                emit_fn=self._emit_and_persist,
                update_freq=max(100, ckpt_freq // 20),
                job_id=self.job_id,
                initial_timesteps=initial_ts,
            )
            # job_id_ref: variable capturada por el closure del callback
            job_id_ref = self.job_id
            # Crear directorio de señales
            signals_dir = BASE_DIR / "signals"
            signals_dir.mkdir(parents=True, exist_ok=True)

            # Callback propio de checkpoint — funciona correctamente en reanudación.
            # SB3's CheckpointCallback usa n_calls que se preserva del modelo cargado,
            # lo que hace que save_freq % n_calls nunca dispare. Este callback usa
            # pasos de la sesión actual para disparar siempre correctamente.
            class _SessionCheckpointCallback(BaseCallback):
                def __init__(self, save_freq, save_path, name_prefix="ckpt"):
                    super().__init__(verbose=0)
                    self.save_freq    = save_freq
                    self.save_path    = save_path
                    self.name_prefix  = name_prefix
                    self._session_steps = 0
                    self._step_start    = 0

                def _on_training_start(self):
                    self._step_start = self.num_timesteps
                    os.makedirs(self.save_path, exist_ok=True)

                def _on_step(self) -> bool:
                    self._session_steps = self.num_timesteps - self._step_start

                    # ── Checkpoint automático por frecuencia ──────────────
                    if self._session_steps > 0 and self._session_steps % self.save_freq == 0:
                        self._save(reason="auto")

                    # ── Checkpoint forzado por archivo de señal ───────────
                    # Cualquier proceso puede crear ./data/signals/<job_id>.ckpt
                    # y el trainer lo detecta en el próximo step sin reiniciar
                    signal_file = os.path.join(
                        str(BASE_DIR), "signals", f"{job_id_ref}.ckpt"
                    )
                    if os.path.exists(signal_file):
                        try:
                            os.remove(signal_file)
                            self._save(reason="forced")
                        except Exception:
                            pass

                    return True

                def _save(self, reason="auto"):
                    path = os.path.join(
                        self.save_path,
                        f"{self.name_prefix}_{reason}_{self.num_timesteps}_steps"
                    )
                    self.model.save(path)
                    print(f"[Checkpoint-{reason}] {path}.zip (step {self.num_timesteps:,})")

            checkpoint_cb = _SessionCheckpointCallback(
                save_freq=ckpt_freq,
                save_path=str(ckpt_dir),
                name_prefix="ckpt",
            )

            self._update_job({"status": "training", "message": "Entrenamiento iniciado"})
            model.learn(
                total_timesteps=total,
                callback=[progress_cb, checkpoint_cb],
                reset_num_timesteps=not bool(self.config.get("resume_from", "")),
            )

            # Guardar modelo final
            final_dir = MODELS_DIR / self.job_id
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / "final_model"
            model.save(str(final_path))
            zip_path = str(final_path) + ".zip"

            self._update_job({
                "status":     "completed",
                "progress":   1.0,
                "model_path": zip_path,
                "message":    f"Completado. Modelo en {zip_path}",
            })
            self._emit_and_persist({
                "job_id":     self.job_id,
                "status":     "completed",
                "progress":   1.0,
                "model_path": zip_path,
                "message":    "Entrenamiento completado",
            })

        except Exception as e:
            self.error = str(e)
            self._update_job({"status": "error", "message": str(e)})
            self._emit_and_persist({
                "job_id":  self.job_id,
                "status":  "error",
                "message": str(e),
            })
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _update_job(self, patch: dict):
        """Actualiza el registry en memoria y persiste en disco."""
        if self.jobs_registry is not None and self.job_id in self.jobs_registry:
            self.jobs_registry[self.job_id].update(patch)
            save_jobs_to_disk(self.jobs_registry)

    def _emit_and_persist(self, data: dict):
        """Emite por WebSocket y persiste el estado en disco."""
        self._update_job({k: v for k, v in data.items() if k != "job_id"})
        try:
            self.emit_fn(data)
        except Exception:
            pass