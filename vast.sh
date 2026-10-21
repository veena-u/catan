#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi

python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy
python -c "import torch" 2>/dev/null || python -m pip install --quiet torch
python -m pip install --quiet ./python

python - <<'EOF'
import os, torch, catan
assert torch.cuda.is_available(), "PyTorch cannot see a GPU; pick a CUDA PyTorch image"
print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}, {os.cpu_count()} CPU cores")
print("catan module ok:", catan.ACTIONS, "actions")
EOF
