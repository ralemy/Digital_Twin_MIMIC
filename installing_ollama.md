# Installing Ollama on a remote machine without root access:

```bash
mkdir -p ~/ollama-local
cd ~/ollama-local
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst
tar --zstd -xf ollama.tar.zst
```

or, if `tar --zstd` is not supported:

```bash
# fallback if `tar --zstd` isn't supported
zstd -d ollama.tar.zst -o ollama.tar
tar -xf ollama.tar
```

if download fails:

```bash
curl -fsSL https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-linux-amd64.tar.zst -o ollama.tar.zst
```

add it to path:
```bash
echo 'export PATH="$HOME/ollama-local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

create a repo for models:
```bash
echo 'export OLLAMA_MODELS="$HOME/ollama-local/models"' >> ~/.bashrc
source ~/.bashrc
mkdir -p "$OLLAMA_MODELS"
```

run server manually:
```bash
tmux new -s ollama
OLLAMA_NUM_PARALLEL=4 ollama serve
# Ctrl+B, D to detach — leave this session running
```
pull a model and check
```bash
ollama pull qwen2.5:32b-instruct-q8_0
nvidia-smi   # confirm the ollama process shows up using GPU memory once you run a prompt
```

