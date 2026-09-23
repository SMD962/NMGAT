from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.workflow import predict_main

if __name__ == '__main__':
    predict_main()
