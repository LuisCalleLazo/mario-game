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
BASE_DIR        = Path(os.environ.get("DATA_DIR", "./data"))
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
                 update_freq: int = 500, job_id: str = ""):
        super().__init__(verbose=0)
        self.total_timesteps = int(total_timesteps)
        self.emit_fn    = emit_fn or (lambda d: None)
        self.update_freq = update_freq
        self.job_id     = job_id
        self.start_time = None
        self.last_mean_reward = 0.0

    def _on_training_start(self):
        self.start_time = time.time()
        self._publish({"status": "training", "timestep": 0, "progress": 0.0})

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq == 0:
            progress = self.num_timesteps / self.total_timesteps
            elapsed  = time.time() - self.start_time
            eta_sec  = (elapsed / max(progress, 1e-9)) * (1 - progress)

            if len(self.model.ep_info_buffer) > 0:
                self.last_mean_reward = float(
                    np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                )

            self._publish({
                "status":      "training",
                "timestep":    self.num_timesteps,
                "total":       self.total_timesteps,
                "progress":    round(progress, 4),
                "elapsed_sec": round(elapsed, 1),
                "eta_sec":     round(eta_sec, 1),
                "mean_reward": round(self.last_mean_reward, 2),
            })
        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        self._publish({
            "status":      "saving",
            "timestep":    self.num_timesteps,
            "total":       self.total_timesteps,
            "progress":    1.0,
            "elapsed_sec": round(elapsed, 1),
            "eta_sec":     0,
            "mean_reward": round(self.last_mean_reward, 2),
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
        env_id = f"SuperMarioBros-{c.get('world',1)}-{c.get('level',1)}-v0"
        return load_smb_env(
            env_id,
            self.CROP_DIM,
            n_stack=int(c.get("n_stack", 4)),
            n_skip=int(c.get("n_skip", 4)),
            action_set=c.get("action_set", "SIMPLE"),
        )

    def _build_model(self, env, device: str):
        c = self.config
        resume_path = c.get("resume_from", "").strip()
        if resume_path and os.path.exists(resume_path):
            print(f"[Trainer] Reanudando desde {resume_path} en {device}")
            return PPO.load(resume_path, env=env, device=device)

        return PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=0,
            learning_rate=float(c.get("learning_rate", 3e-4)),
            n_steps=int(c.get("n_steps", 2048)),
            batch_size=int(c.get("batch_size", 64)),
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

            env   = self._build_env()
            model = self._build_model(env, device)

            total     = int(self.config.get("total_timesteps", 100_000))
            ckpt_freq = int(self.config.get("checkpoint_freq", 10_000))
            ckpt_dir  = CHECKPOINTS_DIR / self.job_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            progress_cb = ProgressCallback(
                total_timesteps=total,
                emit_fn=self._emit_and_persist,
                update_freq=max(100, ckpt_freq // 20),
                job_id=self.job_id,
            )
            checkpoint_cb = CheckpointCallback(
                save_freq=ckpt_freq,
                save_path=str(ckpt_dir),
                name_prefix="ckpt",
                verbose=0,
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