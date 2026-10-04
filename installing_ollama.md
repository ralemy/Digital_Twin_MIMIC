# Installing Ollama on a remote machine without root access:

Unpack into the parent of `$DT_OLLAMA_BIN` (set per cluster in
`jobs/setup_bash.sh`; the archive holds `bin/` and `lib/`):

```bash
source jobs/setup_bash.sh      # from the repository base
mkdir -p "$(dirname "$DT_OLLAMA_BIN")"
cd "$(dirname "$DT_OLLAMA_BIN")"
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

Sourcing `jobs/setup_bash.sh` puts `$DT_OLLAMA_BIN` on the PATH and sets
`OLLAMA_MODELS` to `$DT_OLLAMA_MODELS` (where models are downloaded; the repo's
`ollama-models` symlink points there); jobs do it themselves.

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

