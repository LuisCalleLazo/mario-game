"""
app.py
======
API Flask + Flask-SocketIO para la plataforma Mario RL.

REST endpoints:
  POST /api/train               → lanza job
  POST /api/train/<id>/stop     → detiene job
  GET  /api/jobs                → lista jobs (memoria + disco)
  GET  /api/jobs/<id>           → estado de job + checkpoints
  GET  /api/models              → modelos finales guardados
  GET  /api/models/<id>/download → descarga modelo .zip
  GET  /api/checkpoints/<id>    → lista checkpoints de un job
  GET  /api/checkpoints/<id>/<filename> → descarga checkpoint
  POST /api/evaluate            → evalúa modelo headless
  GET  /api/health              → GPU/CPU info

WebSocket (namespace /ws):
  client→server:  join_job   { job_id }
  server→client:  progress   { job_id, status, timestep, progress, eta_sec, … }
"""

import os
import uuid
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file, abort
from flask_socketio import SocketIO, join_room, emit
from flask_cors import CORS

from player import MarioPlayer
from trainer import (
    MarioTrainer, load_jobs_from_disk, save_jobs_to_disk,
    MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR, BASE_DIR,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "mario-secret-42")
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", path="/ws/socket.io" )

# ── Estado global ─────────────────────────────────────────────────────────────
_jobs: dict = load_jobs_from_disk()       # carga jobs previos del disco
_trainers: dict[str, MarioTrainer] = {}
_players: dict[str, MarioPlayer] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_emit_fn(job_id: str):
    def _emit(data: dict):
        if job_id in _jobs:
            _jobs[job_id].update({k: v for k, v in data.items() if k != "job_id"})
            save_jobs_to_disk(_jobs)
        socketio.emit("progress", data, room=job_id, namespace="/ws")
    return _emit

def _list_checkpoints(job_id: str) -> list[dict]:
    ckpt_dir = CHECKPOINTS_DIR / job_id
    if not ckpt_dir.exists():
        return []
    files = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return [{"filename": f.name, "size_mb": round(f.stat().st_size / 1e6, 2),
             "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
            for f in files]

def _job_public(job: dict) -> dict:
    """Devuelve job sin rutas internas sensibles para el frontend."""
    j = dict(job)
    j["checkpoints"] = _list_checkpoints(j.get("job_id", ""))
    return j


# ══════════════════════════════════════════════════════════════════════════════
# REST — Entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def start_training():
    config = request.get_json(force=True)

    world = int(config.get("world", 1))
    level = int(config.get("level", 1))
    if not (1 <= world <= 8 and 1 <= level <= 4):
        return jsonify({"error": "world 1-8 y level 1-4"}), 400

    job_id = str(uuid.uuid4())
    config["job_id"] = job_id

    # Registro inicial
    _jobs[job_id] = {
        "job_id":          job_id,
        "status":          "queued",
        "world":           world,
        "level":           level,
        "action_set":      config.get("action_set", "SIMPLE"),
        "total_timesteps": int(config.get("total_timesteps", 100_000)),
        "created_at":      datetime.now().isoformat(),
        "progress":        0.0,
        "timestep":        0,
        "mean_reward":     0.0,
        "eta_sec":         None,
        "device":          config.get("device_override", "auto"),
        "model_path":      None,
        "message":         "En cola",
    }
    save_jobs_to_disk(_jobs)

    emit_fn = _make_emit_fn(job_id)
    trainer = MarioTrainer(config, emit_fn=emit_fn, jobs_registry=_jobs)
    _trainers[job_id] = trainer
    trainer.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/train/<job_id>/stop", methods=["POST"])
def stop_training(job_id):
    trainer = _trainers.get(job_id)
    if not trainer:
        return jsonify({"error": "job no encontrado"}), 404
    trainer.stop()
    _jobs[job_id]["status"] = "stopped"
    save_jobs_to_disk(_jobs)
    socketio.emit("progress", {"job_id": job_id, "status": "stopped"}, room=job_id, namespace="/ws")
    return jsonify({"job_id": job_id, "status": "stopped"})


# ══════════════════════════════════════════════════════════════════════════════
# REST — Jobs
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    return jsonify([_job_public(j) for j in _jobs.values()])


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "no encontrado"}), 404
    return jsonify(_job_public(job))


# ══════════════════════════════════════════════════════════════════════════════
# REST — Modelos (descarga)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/models", methods=["GET"])
def list_models():
    models = []
    for job_id, job in _jobs.items():
        mp = job.get("model_path")
        if mp and Path(mp).exists():
            p = Path(mp)
            models.append({
                "job_id":     job_id,
                "path":       mp,
                "filename":   p.name,
                "size_mb":    round(p.stat().st_size / 1e6, 2),
                "created":    datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "world":      job.get("world", "?"),
                "level":      job.get("level", "?"),
                "action_set": job.get("action_set", "?"),
                "mean_reward": job.get("mean_reward", 0),
            })
    return jsonify(sorted(models, key=lambda m: m["created"], reverse=True))


@app.route("/api/models/<job_id>/download", methods=["GET"])
def download_model(job_id):
    job = _jobs.get(job_id)
    if not job:
        abort(404)
    mp = job.get("model_path")
    if not mp or not Path(mp).exists():
        abort(404)
    return send_file(
        mp,
        as_attachment=True,
        download_name=f"mario_w{job.get('world',1)}_l{job.get('level',1)}_{job_id[:8]}.zip",
    )


# ══════════════════════════════════════════════════════════════════════════════
# REST — Checkpoints (descarga)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/checkpoints/<job_id>", methods=["GET"])
def list_checkpoints(job_id):
    return jsonify(_list_checkpoints(job_id))


@app.route("/api/checkpoints/<job_id>/<filename>", methods=["GET"])
def download_checkpoint(job_id, filename):
    path = CHECKPOINTS_DIR / job_id / filename
    if not path.exists() or not path.suffix == ".zip":
        abort(404)
    return send_file(
        str(path),
        as_attachment=True,
        download_name=filename,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REST — Evaluación headless
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/evaluate", methods=["POST"])
def evaluate_model():
    import numpy as np
    from gym_utils import load_smb_env, SMB
    from stable_baselines3 import PPO

    body       = request.get_json(force=True)
    model_path = body.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        return jsonify({"error": "model_path inválido"}), 400

    world    = int(body.get("world", 1))
    level    = int(body.get("level", 1))
    episodes = int(body.get("episodes", 3))
    aset     = body.get("action_set", "SIMPLE")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env    = load_smb_env(f"SuperMarioBros-{world}-{level}-v0",
                               [0, 16, 0, 13], action_set=aset)
        model  = PPO.load(model_path, env=env, device=device)
        smb    = SMB(env, model)
        scores = smb.play(episodes=episodes, deterministic=True, render=False)
        env.close()
        return jsonify({
            "mean_score": float(np.mean(scores)),
            "std_score":  float(np.std(scores)),
            "scores":     [float(s) for s in scores],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# REST — Health
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    import torch
    cuda = torch.cuda.is_available()
    return jsonify({
        "status": "ok",
        "cuda":   cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu (RAM only)",
        "jobs_loaded": len(_jobs),
    })


# ══════════════════════════════════════════════════════════════════════════════
# REST — Play / Stream de frames
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/play", methods=["POST"])
def start_play():
    """
    Lanza un episodio de evaluación con streaming de frames por WebSocket.
    Body: { job_id, model_path, world, level, action_set, episodes, fps_cap }
    El cliente debe unirse al room 'play_<play_id>' para recibir los frames.
    """
    import uuid as _uuid
    body       = request.get_json(force=True)
    model_path = body.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        return jsonify({"error": "model_path inválido"}), 400

    play_id    = str(_uuid.uuid4())
    world      = int(body.get("world", 1))
    level      = int(body.get("level", 1))
    action_set = body.get("action_set", "SIMPLE")
    episodes   = int(body.get("episodes", 3))
    fps_cap    = int(body.get("fps_cap", 24))

    def _emit(event, data):
        socketio.emit(event, data, room=f"play_{play_id}", namespace="/ws")

    player = MarioPlayer(
        model_path=model_path,
        world=world, level=level,
        action_set=action_set,
        emit_fn=_emit,
        job_id=play_id,
        episodes=episodes,
        fps_cap=fps_cap,
    )
    _players[play_id] = player
    player.start()
    return jsonify({"play_id": play_id}), 202


@app.route("/api/play/<play_id>/stop", methods=["POST"])
def stop_play(play_id):
    player = _players.get(play_id)
    if not player:
        return jsonify({"error": "no encontrado"}), 404
    player.stop()
    return jsonify({"play_id": play_id, "status": "stopped"})


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket
# ══════════════════════════════════════════════════════════════════════════════

@socketio.on("join_job", namespace="/ws")
def ws_join_job(data):
    job_id = data.get("job_id", "")
    join_room(job_id)
    job = _jobs.get(job_id, {})
    emit("progress", {"job_id": job_id, **job})


@socketio.on("join_play", namespace="/ws")
def ws_join_play(data):
    play_id = data.get("play_id", "")
    join_room(f"play_{play_id}")
    emit("play_status", {"play_id": play_id, "status": "joined"})


@socketio.on("connect", namespace="/ws")
def ws_connect():
    pass


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[App] Jobs cargados desde disco: {len(_jobs)}")
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )