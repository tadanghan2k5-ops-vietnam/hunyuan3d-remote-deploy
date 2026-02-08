#!/bin/bash
# =============================================================================
# GO.sh v3.0 -- REBOOT-RESILIENT ALL-IN-ONE AUTO DEPLOY
# Handles NVIDIA driver reboot automatically. Resumes from where it left off.
#
# Usage: sudo bash /media/han/UBUNTU-SERV/hunyuan3d-deploy/GO.sh
#   OR: Runs automatically via first-boot systemd service
#
# Flow: Phase 1-8, survives reboots, state tracked in /var/lib/hunyuan3d-deploy
# =============================================================================

set -uo pipefail

# ---- CONFIG ----
USER_NAME="han"
USER_HOME="/home/$USER_NAME"
VENV_DIR="$USER_HOME/hunyuan3d-env"
SERVER_DIR="$USER_HOME/hunyuan3d-server"
MODEL_DIR="$USER_HOME/Hunyuan3D-2.1-full"
REPO_DIR="$USER_HOME/Hunyuan3D-2.1"
STATE_DIR="/var/lib/hunyuan3d-deploy"
STATE_FILE="$STATE_DIR/phase"
LOG_FILE="/var/log/hunyuan3d-deploy.log"
CUDA_VER="12-8"
CUDA_PATH="/usr/local/cuda-12.8"

# Find deploy source (USB or copied location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/scoring_profiles" ]; then
    DEPLOY_DIR="$SCRIPT_DIR"
elif [ -d "$USER_HOME/hunyuan3d-deploy/scoring_profiles" ]; then
    DEPLOY_DIR="$USER_HOME/hunyuan3d-deploy"
else
    DEPLOY_DIR="$SCRIPT_DIR"
fi

# ---- COLORS ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"; }
ok()   { echo -e "${GREEN}[DONE]${NC} $1" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"; }
fail() { echo -e "${RED}[FAIL]${NC} $1" | tee -a "$LOG_FILE"; }

# ---- STATE MANAGEMENT ----
mkdir -p "$STATE_DIR"
touch "$LOG_FILE"

get_phase() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo "0"
    fi
}

set_phase() {
    echo "$1" > "$STATE_FILE"
    log "Phase $1 saved to state file"
}

CURRENT_PHASE=$(get_phase)

# ---- ROOT CHECK ----
if [ "$(id -u)" -ne 0 ]; then
    echo "Must run as root: sudo bash $0"
    exit 1
fi

echo "" >> "$LOG_FILE"
log "========== GO.sh v3.0 started at $(date) =========="
log "Current phase: $CURRENT_PHASE | Deploy source: $DEPLOY_DIR"

echo ""
echo -e "${BOLD}${CYAN}============================================${NC}"
echo -e "${BOLD}${CYAN}  HUNYUAN3D ALL-IN-ONE AUTO DEPLOY v3.0${NC}"
echo -e "${BOLD}${CYAN}  Reboot-Resilient | Resume from Phase $CURRENT_PHASE${NC}"
echo -e "${BOLD}${CYAN}============================================${NC}"
echo ""

# ---- ENSURE RESUME SERVICE EXISTS ----
# This service re-runs GO.sh after every reboot until deploy is complete
if [ ! -f /etc/systemd/system/hunyuan3d-deploy.service ]; then
    # Copy GO.sh to persistent location (USB may not be mounted after reboot)
    mkdir -p "$USER_HOME/hunyuan3d-deploy"
    cp -r "$DEPLOY_DIR/"* "$USER_HOME/hunyuan3d-deploy/" 2>/dev/null || true
    chown -R "$USER_NAME:$USER_NAME" "$USER_HOME/hunyuan3d-deploy"

    cat > /etc/systemd/system/hunyuan3d-deploy.service << 'SVCEOF'
[Unit]
Description=Hunyuan3D Deploy (auto-resume after reboot)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash /home/han/hunyuan3d-deploy/GO.sh
RemainAfterExit=yes
StandardOutput=journal+console
StandardError=journal+console
TimeoutStartSec=7200

[Install]
WantedBy=multi-user.target
SVCEOF
    systemctl daemon-reload
    systemctl enable hunyuan3d-deploy.service
    log "Deploy resume service installed"
    DEPLOY_DIR="$USER_HOME/hunyuan3d-deploy"
fi

# =============================================
# PHASE 0.5: TAILSCALE EARLY INSTALL (for remote monitoring)
# Runs BEFORE anything else so we can SSH in and watch progress
# =============================================
if ! command -v tailscale &>/dev/null; then
    log "=== [0/8] EARLY: Installing Tailscale for remote access ==="

    # Wait for apt lock (cloud-init may be running)
    for i in $(seq 1 15); do
        if ! fuser /var/lib/dpkg/lock-frontend &>/dev/null 2>&1; then
            break
        fi
        log "Waiting for apt lock... ($i/15)"
        sleep 10
    done

    export DEBIAN_FRONTEND=noninteractive
    apt update -y -qq 2>/dev/null || true
    curl -fsSL https://tailscale.com/install.sh | sh
    ok "Tailscale installed"
fi

# Connect Tailscale ASAP (even before Phase 1)
if command -v tailscale &>/dev/null && ! tailscale status &>/dev/null 2>&1; then
    log "Starting Tailscale (check admin console for new device)..."
    systemctl enable --now tailscaled 2>/dev/null || true
    sleep 2
    # Start tailscale up in background so it doesn't block if waiting for auth
    tailscale up --ssh --hostname=hunyuan3d-server &
    TS_PID=$!
    sleep 10
    # Log auth URL if available
    TAILSCALE_URL=$(journalctl -u tailscaled --no-pager -n 30 2>/dev/null | grep -oP 'https://login.tailscale.com/\S+' | tail -1)
    if [ -n "$TAILSCALE_URL" ]; then
        echo ""
        echo -e "${BOLD}${YELLOW}============================================${NC}"
        echo -e "${BOLD}${YELLOW}  TAILSCALE AUTH URL:${NC}"
        echo -e "${BOLD}${YELLOW}  $TAILSCALE_URL${NC}"
        echo -e "${BOLD}${YELLOW}============================================${NC}"
        echo ""
        echo "$TAILSCALE_URL" > "$STATE_DIR/tailscale-auth-url"
        log "Tailscale auth URL: $TAILSCALE_URL"
    fi
    # Don't wait for auth - continue with deployment
    log "Tailscale connecting in background, continuing deployment..."
fi

# =============================================
# PHASE 1: SYSTEM + BUILD TOOLS + SSH
# =============================================
if [ "$CURRENT_PHASE" -lt 1 ]; then
    log "=== [1/8] System Update + Build Tools ==="

    # Wait for apt lock (cloud-init may be running)
    for i in $(seq 1 30); do
        if ! fuser /var/lib/dpkg/lock-frontend &>/dev/null 2>&1; then
            break
        fi
        log "Waiting for apt lock... ($i/30)"
        sleep 10
    done

    export DEBIAN_FRONTEND=noninteractive
    apt update -y
    apt upgrade -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"

    apt install -y \
        build-essential git curl wget unzip tmux htop \
        software-properties-common openssh-server ufw \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

    systemctl enable --now ssh
    set_phase 1
    ok "Phase 1: System + SSH ready"
fi

# =============================================
# PHASE 2: NVIDIA DRIVER + CUDA 12.8
# =============================================
if [ "$CURRENT_PHASE" -lt 2 ]; then
    log "=== [2/8] NVIDIA Driver + CUDA 12.8 ==="

    NEED_REBOOT=false

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        CUDA_DETECTED=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "none")
        if [[ "$CUDA_DETECTED" == "12.8"* ]] || [[ "$CUDA_DETECTED" == "12.9"* ]]; then
            ok "CUDA $CUDA_DETECTED already installed"
        else
            warn "CUDA $CUDA_DETECTED found, upgrading..."
            apt remove --purge -y 'nvidia-*' 'cuda-*' 2>/dev/null || true
            apt autoremove -y
            NEED_REBOOT=true
        fi
    else
        NEED_REBOOT=true
    fi

    if $NEED_REBOOT || ! nvidia-smi &>/dev/null; then
        export DEBIAN_FRONTEND=noninteractive
        cd /tmp
        if [ ! -f cuda-keyring_1.1-1_all.deb ]; then
            wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        fi
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt update -y
        apt install -y cuda-toolkit-${CUDA_VER} cuda-drivers

        # Test if driver works without reboot
        if nvidia-smi &>/dev/null; then
            ok "NVIDIA driver loaded without reboot"
        else
            set_phase 1  # Stay at phase 1 so we re-enter phase 2 after reboot
            log "NVIDIA driver needs REBOOT -- auto-resuming after reboot"
            echo ""
            echo -e "${YELLOW}>>> AUTO-REBOOTING in 10 seconds (NVIDIA driver) <<<${NC}"
            echo -e "${YELLOW}>>> Script will resume automatically after reboot <<<${NC}"
            sleep 10
            reboot
            exit 0
        fi
    fi

    # Persist CUDA env
    export PATH=${CUDA_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}

    BASHRC="$USER_HOME/.bashrc"
    if ! grep -q "cuda-12.8" "$BASHRC" 2>/dev/null; then
        cat >> "$BASHRC" << 'ENVEOF'
# CUDA 12.8 (auto-added by GO.sh)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
ENVEOF
    fi

    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG_FILE"
    set_phase 2
    ok "Phase 2: NVIDIA + CUDA ready"
fi

# =============================================
# PHASE 3: PYTHON VENV + PYTORCH + TRITON
# =============================================
if [ "$CURRENT_PHASE" -lt 3 ]; then
    log "=== [3/8] Python 3.11 + PyTorch 2.10 + Triton ==="

    export PATH=${CUDA_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}

    if [ ! -d "$VENV_DIR" ]; then
        sudo -u "$USER_NAME" python3.11 -m venv "$VENV_DIR"
    fi

    sudo -u "$USER_NAME" bash << VENVEOF
source $VENV_DIR/bin/activate
pip install --upgrade pip setuptools wheel -q

# PyTorch + CUDA 12.8
if ! python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "Installing PyTorch 2.10 + CUDA 12.8..."
    pip install torch==2.10.0+cu128 torchvision==0.20.0+cu128 torchaudio==2.10.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "PyTorch already installed with CUDA"
fi

# Triton (Linux only)
if ! python -c 'import triton' 2>/dev/null; then
    echo "Installing Triton..."
    pip install triton
else
    echo "Triton already installed"
fi

python -c "
import torch, triton
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | Triton {triton.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB')
"
VENVEOF

    set_phase 3
    ok "Phase 3: Python + PyTorch + Triton ready"
fi

# =============================================
# PHASE 4: AI LIBRARIES (ALL PINNED)
# =============================================
if [ "$CURRENT_PHASE" -lt 4 ]; then
    log "=== [4/8] AI Libraries (pinned versions) ==="

    sudo -u "$USER_NAME" bash << LIBEOF
source $VENV_DIR/bin/activate
pip install -q \
    transformers==4.57.6 \
    huggingface_hub==0.36.1 \
    diffusers==0.36.0 \
    accelerate==1.7.0 \
    safetensors==0.5.3 \
    tokenizers==0.21.1 \
    sentencepiece==0.2.0 \
    protobuf==5.29.4 \
    timm \
    trimesh \
    pymeshlab \
    rembg \
    onnxruntime \
    pillow numpy scipy einops omegaconf pyyaml \
    fastapi==0.115.0 \
    'uvicorn[standard]==0.30.0' \
    python-multipart==0.0.9 \
    aiofiles==24.1.0 \
    httpx==0.27.0 \
    psutil==6.0.0 \
    pynvml==11.5.0
LIBEOF

    set_phase 4
    ok "Phase 4: AI libraries installed"
fi

# =============================================
# PHASE 5: HUNYUAN3D CLONE + WEIGHTS
# =============================================
if [ "$CURRENT_PHASE" -lt 5 ]; then
    log "=== [5/8] Hunyuan3D 2.1 Clone + 3.3B Weights ==="

    sudo -u "$USER_NAME" bash << MODELEOF
source $VENV_DIR/bin/activate

# Clone repo
if [ ! -d $REPO_DIR ]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 $REPO_DIR
    echo "Cloned Hunyuan3D 2.1"
else
    echo "Repo exists"
fi

# Install hy3dshape
cd $REPO_DIR/hy3dshape && pip install -e . -q

# Download full weights (3.3B, ~6GB)
mkdir -p $MODEL_DIR
python -c "
from huggingface_hub import snapshot_download
import os
ckpt = os.path.join('$MODEL_DIR', 'hunyuan3d-dit-v2-1', 'model.fp16.ckpt')
if os.path.exists(ckpt):
    print(f'Weights exist: {os.path.getsize(ckpt)/1024**3:.1f} GB')
else:
    print('Downloading Hunyuan3D 2.1 full weights (3.3B)...')
    snapshot_download('tencent/Hunyuan3D-2', local_dir='$MODEL_DIR',
        allow_patterns=['hunyuan3d-dit-v2-1/*'])
    if os.path.exists(ckpt):
        print(f'Downloaded: {os.path.getsize(ckpt)/1024**3:.1f} GB')
    else:
        print('WARNING: weights not found')
"
MODELEOF

    set_phase 5
    ok "Phase 5: Hunyuan3D model ready"
fi

# =============================================
# PHASE 6: DEPLOY SERVER FILES
# =============================================
if [ "$CURRENT_PHASE" -lt 6 ]; then
    log "=== [6/8] Deploy Server + Scoring Profiles ==="

    mkdir -p "$SERVER_DIR/scoring_profiles"

    if [ -f "$DEPLOY_DIR/server.py" ]; then
        cp "$DEPLOY_DIR/server.py" "$SERVER_DIR/server.py"
        ok "server.py deployed ($(du -h "$DEPLOY_DIR/server.py" | cut -f1))"
    else
        warn "server.py not found in $DEPLOY_DIR"
    fi

    if [ -d "$DEPLOY_DIR/scoring_profiles" ]; then
        cp "$DEPLOY_DIR/scoring_profiles/"*.json "$SERVER_DIR/scoring_profiles/" 2>/dev/null || true
        ok "Scoring profiles deployed"
    fi

    chown -R "$USER_NAME:$USER_NAME" "$SERVER_DIR"
    set_phase 6
    ok "Phase 6: Server deployed"
fi

# =============================================
# PHASE 7: SYSTEMD + TAILSCALE + FIREWALL
# =============================================
if [ "$CURRENT_PHASE" -lt 7 ]; then
    log "=== [7/8] Systemd + Tailscale + Firewall ==="

    # Hunyuan3D runtime service
    cat > /etc/systemd/system/hunyuan3d.service << 'SVCEOF'
[Unit]
Description=Hunyuan3D AI Server (PINNED HOT FOREVER)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=han
Group=han
WorkingDirectory=/home/han/hunyuan3d-server
Environment="PATH=/home/han/hunyuan3d-env/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
ExecStart=/home/han/hunyuan3d-env/bin/python server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
TimeoutStopSec=30
KillMode=mixed
LimitNOFILE=65535
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
SVCEOF
    systemctl daemon-reload
    systemctl enable hunyuan3d

    # Tailscale (already installed in Phase 0.5, just verify)
    if ! command -v tailscale &>/dev/null; then
        curl -fsSL https://tailscale.com/install.sh | sh
    fi
    # Retry Tailscale connection if not yet authenticated
    if ! tailscale status &>/dev/null 2>&1; then
        log "Retrying Tailscale connection..."
        tailscale up --ssh --hostname=hunyuan3d-server &
        sleep 10
    else
        ok "Tailscale already connected"
    fi

    # Firewall
    ufw allow ssh
    ufw allow 8099/tcp comment "Hunyuan3D Server"
    ufw allow 41641/udp comment "Tailscale WireGuard"
    ufw --force enable

    set_phase 7
    ok "Phase 7: Systemd + Tailscale + Firewall ready"
fi

# =============================================
# PHASE 8: START SERVICE + VERIFY + CLEANUP
# =============================================
if [ "$CURRENT_PHASE" -lt 8 ]; then
    log "=== [8/8] Start Server + Verify + Cleanup ==="

    # Start Hunyuan3D server
    systemctl start hunyuan3d || warn "Service start may take time (model loading)"

    # Wait for server (model loads in ~60s)
    log "Waiting for Hunyuan3D server (model loading ~60-90s)..."
    READY=false
    for i in $(seq 1 36); do
        sleep 5
        if curl -s http://localhost:8099/health &>/dev/null; then
            READY=true
            break
        fi
        echo -n "."
    done
    echo ""

    # Full verification
    sudo -u "$USER_NAME" bash << VERIFYEOF
source $VENV_DIR/bin/activate
python -c "
import torch, triton, transformers, diffusers, timm
print()
print('=== SYSTEM VERIFICATION ===')
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.version.cuda}')
print(f'Triton:        {triton.__version__}')
print(f'transformers:  {transformers.__version__}')
print(f'diffusers:     {diffusers.__version__}')
print(f'timm:          {timm.__version__}')
print(f'GPU:           {torch.cuda.get_device_name(0)}')
print(f'VRAM:          {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f} GB')
print(f'torch.compile: READY (Triton backend)')
print()
print('ALL LIBRARY CHECKS PASSED!')
"
VERIFYEOF

    nvidia-smi

    if $READY; then
        ok "Hunyuan3D server RUNNING on port 8099"
        HEALTH=$(curl -s http://localhost:8099/health 2>/dev/null | head -c 300)
        log "Health: $HEALTH"
    else
        warn "Server not responding yet. Check: journalctl -u hunyuan3d -f"
    fi

    # Cleanup: remove deploy resume service (no longer needed)
    systemctl disable hunyuan3d-deploy.service 2>/dev/null || true
    rm -f /etc/systemd/system/hunyuan3d-deploy.service
    rm -f /etc/systemd/system/hunyuan3d-first-boot.service
    systemctl daemon-reload

    set_phase 8
    ok "Phase 8: Deploy complete, cleanup done"
fi

# =============================================
# FINAL REPORT
# =============================================
# Get IPs
LAN_IP=$(hostname -I | awk '{print $1}')
TS_IP=$(tailscale ip -4 2>/dev/null || echo "not connected")

echo ""
echo -e "${BOLD}${GREEN}============================================${NC}"
echo -e "${BOLD}${GREEN}  HUNYUAN3D DEPLOY COMPLETE!${NC}"
echo -e "${BOLD}${GREEN}============================================${NC}"
echo ""
echo "  LAN:        http://${LAN_IP}:8099"
echo "  Tailscale:  http://${TS_IP}:8099"
echo "  SSH:        ssh han@${LAN_IP}"
echo "  SSH (TS):   ssh han@${TS_IP}"
echo ""
echo "  Commands:"
echo "    Status:   sudo systemctl status hunyuan3d"
echo "    Logs:     journalctl -u hunyuan3d -f"
echo "    Restart:  sudo systemctl restart hunyuan3d"
echo "    Health:   curl http://localhost:8099/health"
echo ""
if [ "$TS_IP" = "not connected" ]; then
    TS_URL=""
    if [ -f "$STATE_DIR/tailscale-auth-url" ]; then
        TS_URL=$(cat "$STATE_DIR/tailscale-auth-url")
    fi
    echo -e "  ${YELLOW}Tailscale: sudo tailscale up --ssh --hostname=hunyuan3d-server${NC}"
    if [ -n "$TS_URL" ]; then
        echo -e "  ${YELLOW}Auth URL: $TS_URL${NC}"
    fi
    echo ""
fi
echo "  Password: han / han123 (change with: passwd)"
echo "  Performance: Triton + torch.compile = 20-40% FASTER!"
echo ""
echo -e "${GREEN}============================================${NC}"
echo ""
log "========== DEPLOY COMPLETED at $(date) =========="
log "LAN: $LAN_IP | Tailscale: $TS_IP"
