from pathlib import Path

# Root directory of the application (parent of this ui/ package).
# Use APP_DIR / 'filename' instead of bare 'filename' so paths are stable
# regardless of the working directory at launch time.
APP_DIR = Path(__file__).parent.parent
