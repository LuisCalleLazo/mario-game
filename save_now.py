# save_now.py
# Ejecutar con: sudo gdb -p PID -batch -ex 'call PyRun_SimpleString("exec(open(\"save_now.py\").read())")'
# O desde pyrasite: pyrasite-shell PID -> exec(open("save_now.py").read())

import os, sys, gc
from pathlib import Path

JOB_ID   = "0c2f9af9-36f4-40aa-8d33-b247ce0add51"
SAVE_DIR = Path("./data/checkpoints") / JOB_ID
SAVE_DIR.mkdir(parents=True, exist_ok=True)

saved = False

# Buscar el modelo en todos los objetos vivos en memoria
for obj in gc.get_objects():
    try:
        # Stable Baselines3 PPO tiene atributo 'policy' y 'env'
        if hasattr(obj, 'policy') and hasattr(obj, 'num_timesteps') and hasattr(obj, 'save'):
            ts   = obj.num_timesteps
            path = str(SAVE_DIR / f"ckpt_emergency_{ts}_steps")
            obj.save(path)
            print(f"[GUARDADO] {path}.zip  (step {ts:,})")
            saved = True
    except Exception as e:
        pass

if not saved:
    print("[ERROR] No se encontró ningún modelo PPO en memoria")
else:
    print("[OK] Modelo guardado exitosamente")