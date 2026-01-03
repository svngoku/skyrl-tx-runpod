# SkyRL TX RunPod Setup

Portable setup script for deploying SkyRL TX (Transfer Learning API) on RunPod GPU instances. The script auto-detects GPU count and VRAM to configure optimal settings across H100/H200/B200 GPUs.

## Features

- **Auto-detection**: Automatically detects GPU count and VRAM via `nvidia-smi`
- **Portable**: Works on single-GPU or multi-GPU setups
- **Configurable**: Override defaults via environment variables
- **Tmux integration**: Server runs in detached tmux session
- **Health checks**: Waits for server to be ready before continuing

## Quick Start on RunPod

1. Copy `setup_skyrl_tx_runpod.sh` to your RunPod instance
2. Make it executable and run:

```bash
chmod +x setup_skyrl_tx_runpod.sh
./setup_skyrl_tx_runpod.sh
```

The script will:
- Detect your GPUs and configure optimal settings
- Install dependencies (uv, system packages)
- Clone the SkyRL repository
- Start the SkyRL TX server in a tmux session

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-4B` | Base model to use |
| `PORT` | `8000` | API server port |
| `TP_SIZE` | `auto` (GPU count) | Tensor parallel size |
| `TRAIN_MICRO_BS` | `auto` (VRAM-based) | Training micro-batch size |
| `MAX_LORA_ADAPTERS` | `3` | Maximum LoRA adapters |
| `MAX_LORA_RANK` | `1` | Maximum LoRA rank |
| `WORKDIR` | `/workspace` | Working directory |
| `HF_TOKEN` | (empty) | HuggingFace token for private models |
| `WANDB_API_KEY` | (empty) | Weights & Biases API key |
| `RUN_RL_LOOP` | `0` | Set to `1` to run RL loop |

### GPU-Aware Defaults

The script automatically chooses `TRAIN_MICRO_BS` based on minimum GPU VRAM:

| VRAM | Micro-Batch Size |
|------|------------------|
| >= 180 GB (B200) | 16 |
| >= 120 GB (H200) | 12 |
| >= 80 GB (H100) | 8 |
| < 80 GB | 4 |

## Usage Examples

### Default (auto-detect all GPUs)
```bash
./setup_skyrl_tx_runpod.sh
```

### Single GPU configuration
```bash
TP_SIZE=1 TRAIN_MICRO_BS=4 ./setup_skyrl_tx_runpod.sh
```

### Custom model and port
```bash
MODEL=meta-llama/Llama-3.1-8B PORT=9000 ./setup_skyrl_tx_runpod.sh
```

### With HuggingFace token (for gated models)
```bash
HF_TOKEN=hf_xxx MODEL=meta-llama/Llama-3.1-70B ./setup_skyrl_tx_runpod.sh
```

### Run RL loop after setup
```bash
RUN_RL_LOOP=1 WANDB_API_KEY=xxx ./setup_skyrl_tx_runpod.sh
```

## Monitoring

### Attach to server session
```bash
tmux attach -t skyrl-tx
```
Press `Ctrl+B` then `D` to detach without stopping.

### View logs
```bash
tail -f /workspace/skyrl_tx_out.log
```

### Check server status
```bash
curl http://localhost:8000/health
```

## Requirements

- NVIDIA GPU(s) with `nvidia-smi` available
- Linux-based system (Ubuntu/Debian recommended)
- Internet connection for cloning repos and installing packages

## RunPod Deployment

### Template Settings

Recommended RunPod template settings:

**Base Image**: `runpod/base:latest` or any Ubuntu-based PyTorch image

**Container Disk**: Minimum 50GB

**Volume**: Mount a network volume for persistence (optional but recommended)

**Environment Variables** (in RunPod template):
```
HF_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_key_here
```

### Deploying

1. Create a new RunPod pod with a GPU template (H100/H200/B200)
2. SSH into the pod
3. Download and run the script:

```bash
curl -O https://raw.githubusercontent.com/svngoku/skyrl-tx-runpod/main/setup_skyrl_tx_runpod.sh
chmod +x setup_skyrl_tx_runpod.sh
./setup_skyrl_tx_runpod.sh
```

## Troubleshooting

### Port already in use
```bash
# Check what's using the port
lsof -i :8000
# Or use a different port
PORT=9000 ./setup_skyrl_tx_runpod.sh
```

### Server not starting
```bash
# Check the logs
tail -n 200 /workspace/skyrl_tx_out.log
# Or attach to the tmux session
tmux attach -t skyrl-tx
```

### GPU not detected
Ensure `nvidia-smi` works:
```bash
nvidia-smi
```

## Resources

- [SkyRL Repository](https://github.com/NovaSky-AI/SkyRL)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [RunPod Documentation](https://docs.runpod.io/)
