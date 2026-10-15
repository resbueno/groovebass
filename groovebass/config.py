import os
from pathlib import Path

from kicksnare.ingest import DATA_DIR as KICKSNARE_DATA

# Dados próprios (banco e uma pasta por música); o Kicksnare fica em KICKSNARE_DATA
DATA_DIR = Path(os.environ.get("GROOVEBASS_DATA") or KICKSNARE_DATA / "groovebass")
BASE = os.environ.get("GROOVEBASS_BASE", "/groovebass").rstrip("/")
UPLOADS = DATA_DIR / "uploads"
MAX_UPLOAD = 300 * 1024 * 1024
MAX_SONGS = int(os.environ.get("GROOVEBASS_MAX_SONGS", "25"))     # por usuário
MAX_ACTIVE = int(os.environ.get("GROOVEBASS_MAX_ACTIVE", "3"))    # na fila + processando, por usuário
MAX_QUEUE_TOTAL = int(os.environ.get("GROOVEBASS_MAX_QUEUE", "20"))
MIN_FREE_GB = float(os.environ.get("KICKSNARE_MIN_FREE_GB", "15"))
