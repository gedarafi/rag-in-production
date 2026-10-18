#!/usr/bin/env bash
# Let Docker use the NVIDIA GPU (NVIDIA Container Toolkit), so LocalStack's Bedrock model runs on it.
#
# LocalStack starts its Ollama container itself and accepts no --gpus or --runtime flag for it, so the
# nvidia runtime becomes Docker's default. It only hands a GPU to containers that ask for one
# (NVIDIA_VISIBLE_DEVICES, which `just up` sets for LocalStack's model container); other containers
# run as before. Debian/Ubuntu (incl. Pop!_OS) only; needs sudo and an installed NVIDIA driver. Safe
# to run again. Restarting Docker stops LocalStack and Phoenix (LocalStack loses its state; run
# `just local-up` afterwards).
set -euo pipefail

command -v nvidia-smi >/dev/null || { echo "No NVIDIA driver found (nvidia-smi); install it first." >&2; exit 1; }

keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sources=/etc/apt/sources.list.d/nvidia-container-toolkit.list

# apt also finishes any other half-installed packages (e.g. a pending kernel), so skip it when the
# toolkit is already there.
if dpkg-query -W -f='${Status}' nvidia-container-toolkit 2>/dev/null | grep -q "install ok installed"; then
  echo "The NVIDIA Container Toolkit is already installed."
else
  if [ ! -f "$sources" ]; then
    echo "Adding NVIDIA's apt repository..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o "$keyring"
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=$keyring] https://#" \
      | sudo tee "$sources" >/dev/null
  fi
  echo "Installing the NVIDIA Container Toolkit..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
fi

echo "Making nvidia Docker's default runtime and restarting Docker..."
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

echo "Checking that a container sees the GPU without extra flags, as LocalStack's will..."
[ "$(docker info --format '{{.DefaultRuntime}}')" = nvidia ] || { echo "Docker's default runtime isn't nvidia." >&2; exit 1; }
docker run --rm -e NVIDIA_VISIBLE_DEVICES=all ubuntu nvidia-smi
echo "Done: Docker can use the GPU. Run \`just local-up <email>\` to start the local stack on it."
