import os
import sys
from pathlib import Path

# Add project root and backend folder to Python path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
