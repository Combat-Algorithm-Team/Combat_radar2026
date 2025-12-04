# 在 Windows 笔记本（WSL2 + RTX4060）上运行 GPU 容器指南

本指南帮你在 Windows 机器（RTX4060）通过 WSL2 + Docker 运行项目 GPU 版本镜像，并启用 CUDA/TensorRT。

## 前提条件
- Windows 10/11 安装 WSL2（Ubuntu 22.04 建议）
- 安装最新 NVIDIA 显卡驱动（Windows 端）
- 安装 Docker Desktop，并启用 WSL2 集成
- 在 WSL2 发行版中安装 `nvidia-container-toolkit`

### 安装 nvidia-container-toolkit（在 WSL2 Ubuntu 中）
```bash
# WSL2 Ubuntu 终端内执行
sudo apt-get update
sudo apt-get install -y curl gnupg2 ca-certificates
curl -fsSL https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker || true
```

## 构建 GPU 镜像
在仓库根目录中（WSL2 Ubuntu 终端）：
```bash
# 可选：代理（如使用 clash）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 构建镜像（名称：radar-dev:gpu）
docker build -f dockerfile.gpu -t radar-dev:gpu \
  --build-arg HTTP_PROXY="$HTTP_PROXY" \
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
  .
```

## 运行 GPU 容器
```bash
docker run --rm -it \
  --gpus all \
  --net host \
  --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$(pwd)":/workspace \
  -w /workspace \
  radar-dev:gpu
```

可选（GUI）：若需显示 PyQt5 GUI，需在 Windows 端启动 X Server（如 X410/vcXsrv），并在容器传递 DISPLAY：
```bash
# 假设 Windows X Server 使用 :0 并允许网络访问
export DISPLAY=:0

docker run --rm -it \
  --gpus all \
  --net host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v "$(pwd)":/workspace \
  radar-dev:gpu
```

## 简单验证
进入容器后：
```bash
# 1) CUDA
nvidia-smi

# 2) PyTorch CUDA
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
print('current device:', torch.cuda.current_device())
print('name:', torch.cuda.get_device_name(0))
PY

# 3) TensorRT（可选）
python - << 'PY'
import tensorrt as trt
print('TensorRT version:', trt.__version__)
PY
```

## 在 VS Code Dev Containers 中使用（可选）
- 打开仓库，安装 "Dev Containers" 扩展。
- 修改 `.devcontainer/devcontainer.json` 的 `image` 为 `radar-dev:gpu`（或新增一个 gpu 版 devcontainer）。
- 选择 "Reopen in Container"，确保 Docker Desktop + WSL2 GPU 已就绪。

示例 devcontainer 片段：
```json
{
  "name": "RM Radar Dev (GPU)",
  "image": "radar-dev:gpu",
  "workspaceFolder": "/workspace",
  "runArgs": [
    "--gpus", "all",
    "--net", "host",
    "--privileged"
  ],
  "containerEnv": {
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"
  }
}
```

## 常见问题
- `nvidia-smi` 报错：检查 Windows 端显卡驱动与 Docker Desktop 的 WSL2 集成是否开启；确认 WSL2 中安装了 `nvidia-container-toolkit` 并重启 Docker。
- `torch.cuda.is_available() == False`：确保使用了 `--gpus all` 并且镜像内 PyTorch 是 CUDA 版（本 Dockerfile 使用 cu121 channel）。
- TensorRT 版本兼容：若 `tensorrt` 安装失败或不兼容，可临时跳过 TensorRT，仅使用 PyTorch + ONNX Runtime GPU；或调整版本到与 CUDA 12.1 兼容的 wheel。
- GUI 无法显示：Windows X Server 未启动或 DISPLAY/防火墙未配置；在 WSL2 下 GUI 体验不稳定，建议主要用于计算与调试，生产环境用 Linux 原生。