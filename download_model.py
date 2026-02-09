from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-1.7B"
local_dir = "models/Qwen3-1.7B"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
