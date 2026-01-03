#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Pretty CLI output
# -----------------------
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"
  DIM="$(tput dim || true)"
  RESET="$(tput sgr0)"

  RED="$(tput setaf 1)"
  GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"
  MAGENTA="$(tput setaf 5)"
  CYAN="$(tput setaf 6)"
else
  BOLD=""; DIM=""; RESET=""
  RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
fi

ts() { date +"%H:%M:%S"; }
log()  { printf "%s %b\n" "$(ts)" "$*"; }
info() { log "${BLUE}${BOLD}[INFO]${RESET} $*"; }
ok()   { log "${GREEN}${BOLD}[OK]${RESET}   $*"; }
warn() { log "${YELLOW}${BOLD}[WARN]${RESET} $*"; }
err()  { log "${RED}${BOLD}[ERR]${RESET}  $*" >&2; }
step() { log "${MAGENTA}${BOLD}==>${RESET} $*"; }

trap 'err "Command failed (exit=$?): ${BASH_COMMAND}"' ERR

# -----------------------
# User-configurable knobs
# -----------------------
WORKDIR="${WORKDIR:-/workspace}"
PORT="${PORT:-8000}"

MODEL="${MODEL:-Qwen/Qwen3-4B}"

# Overrides (optional):
#   TP_SIZE=...            # default: auto-detected GPU count
#   TRAIN_MICRO_BS=...     # default: chosen from VRAM heuristics
#   MAX_LORA_ADAPTERS=...
#   MAX_LORA_RANK=...
TP_SIZE="${TP_SIZE:-auto}"
TRAIN_MICRO_BS="${TRAIN_MICRO_BS:-auto}"

MAX_LORA_ADAPTERS="${MAX_LORA_ADAPTERS:-3}"
MAX_LORA_RANK="${MAX_LORA_RANK:-1}"

# Optional: set to 1 if you also want to run the cookbook RL loop on the same machine
RUN_RL_LOOP="${RUN_RL_LOOP:-0}"

# Optional but recommended for private models / higher HF limits
export HF_TOKEN="${HF_TOKEN:-}"

# Optional for the RL loop example
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export TINKER_API_KEY="${TINKER_API_KEY:-dummy}"

# -----------------------
# Helpers
# -----------------------
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }
}

is_int() { [[ "${1:-}" =~ ^[0-9]+$ ]]; }

detect_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found. This script expects an NVIDIA GPU machine."
    exit 1
  fi

  # Count GPUs (one line per GPU)
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if ! is_int "$GPU_COUNT" || [ "$GPU_COUNT" -lt 1 ]; then
    err "No GPUs detected via nvidia-smi -L."
    exit 1
  fi

  # Query name + memory (MiB) for a simple heuristic.
  # --format=csv,noheader,nounits gives machine-readable output.
  # Example line: "NVIDIA H100 80GB HBM3, 81251"
  mapfile -t GPU_QUERY < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)

  # GPU names (unique)
  GPU_NAMES_UNIQ="$(printf "%s\n" "${GPU_QUERY[@]}" \
    | awk -F',' '{gsub(/^ +| +$/, "", $1); print $1}' \
    | sort -u | tr '\n' ';' | sed 's/;$//')"

  # Min memory across GPUs (MiB) to be conservative
  GPU_MEM_MIN_MIB="$(printf "%s\n" "${GPU_QUERY[@]}" \
    | awk -F',' '{gsub(/^ +| +$/, "", $2); print $2}' \
    | sort -n | head -n1 | tr -d ' ')"
  if ! is_int "$GPU_MEM_MIN_MIB"; then
    GPU_MEM_MIN_MIB=0
  fi

  # Basic info
  info "Detected GPUs: count=${GPU_COUNT}"
  info "GPU models: ${GPU_NAMES_UNIQ:-unknown}"
  if [ "$GPU_MEM_MIN_MIB" -gt 0 ]; then
    info "Min GPU VRAM: ${GPU_MEM_MIN_MIB} MiB (conservative)"
  fi
}

choose_defaults() {
  # TP_SIZE default: all GPUs
  if [ "${TP_SIZE}" = "auto" ]; then
    TP_SIZE="$GPU_COUNT"
  fi

  if ! is_int "$TP_SIZE" || [ "$TP_SIZE" -lt 1 ]; then
    err "Invalid TP_SIZE=${TP_SIZE}. Must be a positive integer."
    exit 1
  fi

  # TRAIN_MICRO_BS heuristic based on min VRAM (MiB)
  if [ "${TRAIN_MICRO_BS}" = "auto" ]; then
    if [ "$GPU_MEM_MIN_MIB" -ge 180000 ]; then
      TRAIN_MICRO_BS=16
    elif [ "$GPU_MEM_MIN_MIB" -ge 120000 ]; then
      TRAIN_MICRO_BS=12
    elif [ "$GPU_MEM_MIN_MIB" -ge 80000 ]; then
      TRAIN_MICRO_BS=8
    else
      TRAIN_MICRO_BS=4
    fi
  fi

  if ! is_int "$TRAIN_MICRO_BS" || [ "$TRAIN_MICRO_BS" -lt 1 ]; then
    err "Invalid TRAIN_MICRO_BS=${TRAIN_MICRO_BS}. Must be a positive integer."
    exit 1
  fi

  if [ "$TP_SIZE" -gt "$GPU_COUNT" ]; then
    warn "TP_SIZE=${TP_SIZE} > GPU_COUNT=${GPU_COUNT}. Lowering TP_SIZE to GPU_COUNT."
    TP_SIZE="$GPU_COUNT"
  fi

  ok "Using config: model=${MODEL} port=${PORT} TP_SIZE=${TP_SIZE} TRAIN_MICRO_BS=${TRAIN_MICRO_BS}"
}

banner() {
  log "${CYAN}${BOLD}SkyRL tx Setup (portable)${RESET}"
  log "${DIM}workdir=${WORKDIR} port=${PORT} model=${MODEL}${RESET}"
}

# -----------------------
# Start
# -----------------------
banner

step "Preparing workspace"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# -----------------------
# System dependencies
# -----------------------
if command -v apt-get >/dev/null 2>&1; then
  step "Installing system packages (apt)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3 python3-venv python3-pip \
    build-essential \
    lsof netcat-openbsd tmux
  ok "System packages installed"
else
  warn "apt-get not found; skipping OS package installation"
fi

need_cmd git
need_cmd curl

# -----------------------
# GPU detection + defaults
# -----------------------
step "Detecting GPUs and choosing defaults"
detect_gpus
choose_defaults

# Non-fatal GPU status snapshot
info "nvidia-smi snapshot:"
nvidia-smi || true

# -----------------------
# Install uv (Astral)
# -----------------------
step "Installing uv (if missing)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
need_cmd uv
ok "uv ready: $(uv --version 2>/dev/null || true)"

# -----------------------
# Clone SkyRL
# -----------------------
step "Fetching SkyRL repo"
if [ ! -d "$WORKDIR/SkyRL" ]; then
  git clone https://github.com/NovaSky-AI/SkyRL.git
  ok "Cloned SkyRL"
else
  info "SkyRL already exists; pulling latest"
  (cd "$WORKDIR/SkyRL" && git pull) || warn "git pull failed; continuing with existing checkout"
fi

# -----------------------
# Start SkyRL tx server
# -----------------------
need_cmd tmux
need_cmd lsof
need_cmd nc

cd "$WORKDIR/SkyRL/skyrl-tx"

step "Validating port availability"
if lsof -iTCP -sTCP:LISTEN -P | grep -q ":${PORT} "; then
  err "Port ${PORT} already in use. Set PORT=xxxx and rerun."
  exit 1
fi
ok "Port ${PORT} is free"

SESSION="skyrl-tx"
LOGFILE="$WORKDIR/skyrl_tx_out.log"

step "Starting SkyRL tx in tmux session: ${SESSION}"
tmux has-session -t "$SESSION" 2>/dev/null && { warn "Killing existing tmux session: ${SESSION}"; tmux kill-session -t "$SESSION"; }

tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd '$WORKDIR/SkyRL/skyrl-tx'
  echo '[tx] launching...'
  uv run --extra gpu --extra tinker -m tx.tinker.api \
    --base-model '$MODEL' \
    --max-lora-adapters '$MAX_LORA_ADAPTERS' \
    --max-lora-rank '$MAX_LORA_RANK' \
    --tensor-parallel-size '$TP_SIZE' \
    --train-micro-batch-size '$TRAIN_MICRO_BS' \
    > '$LOGFILE' 2>&1
"

info "Waiting for SkyRL tx to listen on 127.0.0.1:${PORT} (timeout: 120s) ..."
for i in $(seq 1 120); do
  if nc -z 127.0.0.1 "${PORT}" >/dev/null 2>&1; then
    ok "SkyRL tx is up (port ${PORT})"
    break
  fi
  sleep 1
done

if ! nc -z 127.0.0.1 "${PORT}" >/dev/null 2>&1; then
  err "SkyRL tx did not start within the timeout."
  err "Check logs: tail -n 200 ${LOGFILE}"
  err "Or attach: tmux attach -t ${SESSION}"
  exit 1
fi

info "Attach to server session: tmux attach -t ${SESSION}"
info "Follow logs: tail -f ${LOGFILE}"

# -----------------------
# Optional: run RL loop (tinker-cookbook)
# -----------------------
if [ "$RUN_RL_LOOP" = "1" ]; then
  step "Running RL loop (tinker-cookbook)"
  cd "$WORKDIR"

  if [ ! -d "$WORKDIR/tinker-cookbook" ]; then
    git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
    ok "Cloned tinker-cookbook"
  else
    info "tinker-cookbook already exists; pulling latest"
    (cd "$WORKDIR/tinker-cookbook" && git pull) || warn "git pull failed; continuing with existing checkout"
  fi

  cd "$WORKDIR/tinker-cookbook/recipes"

  if [ -z "${WANDB_API_KEY}" ]; then
    warn "WANDB_API_KEY is empty. The RL loop may fail if wandb is required."
  fi

  info "Starting RL loop against base_url=http://localhost:${PORT}"
  uv run --with wandb --with tinker rl_loop.py \
    base_url="http://localhost:${PORT}" \
    model_name="${MODEL}" \
    lora_rank="${MAX_LORA_RANK}" \
    max_length=1024 \
    save_every=100

  ok "RL loop completed"
fi

ok "Setup complete"
