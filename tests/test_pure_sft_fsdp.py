import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_pure_sft_fsdp_entrypoint_uses_safe_repnoise_with_fsdp():
    source = read("train/train_pure_sft_fsdp.py")
    tree = ast.parse(source)

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {
        node.func.id
        for node in calls
        if isinstance(node.func, ast.Name)
    }

    assert "get_repnoise" in called_names
    assert "build_fsdp_model" in called_names
    assert "SFTTrainer" in called_names
    assert 'safe_data, _ = get_repnoise(split="train")' in source
    assert "unsafe_data" not in source
    assert "save_checkpoint_epoch=args.save_epochs" in source
    assert "spec_from_file_location(" in source
    assert '"reproduce",' in source


def test_pure_sft_fsdp_help_uses_local_reproduce_package():
    result = subprocess.run(
        [sys.executable, "train/train_pure_sft_fsdp.py", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--save-epochs" in result.stdout
    assert "--grad-accum" in result.stdout


def test_run_pure_sft_fsdp_matches_reproduction_script_conventions():
    source = read("scripts/run_pure_sft_fsdp.sh")

    assert 'SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/pure_sft_fsdp}"' in source
    assert 'RUN_NAME="${RUN_NAME:-pure_sft_fsdp}"' in source
    assert "torchrun --standalone --nproc_per_node=\"$GPU_COUNT\"" in source
    assert "train/train_pure_sft_fsdp.py" in source
    assert "--save-epochs" in source


def test_sft_trainer_uses_fsdp_aware_gradient_clipping():
    source = read("train/trainer.py")
    class_start = source.index("class SFTTrainer:")
    next_class = source.index("\nclass RepnoiseTrainer:", class_start)
    sft_source = source[class_start:next_class]

    assert "grad_norm = clip_grad_norm(self.model, self.max_grad_norm)" in sft_source
    assert "torch.nn.utils.clip_grad_norm_(self.model.parameters()" not in sft_source
