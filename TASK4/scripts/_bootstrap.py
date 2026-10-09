from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if (ROOT/'.runtime').is_dir():sys.path.insert(0,str(ROOT/'.runtime'))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
