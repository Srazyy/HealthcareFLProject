# Engineering Rules

## Code Standards

- **Python 3.10+** — use `X | Y` union syntax, not `Union[X, Y]`.
- Type-hint every public function signature (args + return).
- Docstrings on every public function: one-line summary, `Args:`, `Returns:`.
- Use `logging` (module-level `logger = logging.getLogger(__name__)`). Never `print()` in library code; `print()` is acceptable only in `__main__` blocks and notebooks.
- Imports: stdlib → third-party → project-local, separated by blank lines.

## Architecture Invariants

- **Privacy boundary is the client.** Raw text/labels never leave `HealthcareClient`. Only noised LoRA deltas are returned to the server.
- **LoRA-only training.** Base DistilBERT weights stay frozen. Optimizers must filter to `requires_grad` params only.
- **Opacus wrapping order:** `validate_and_fix_model()` → create optimizer → `make_private()`. Reversing this breaks `GradSampleModule`.
- **Noise calibration scope:** `epochs` passed to `make_private()` must equal `local_epochs × num_rounds` (the full training horizon), not a single round.

## Banned Patterns

- No `# TODO` or `// TODO` left in delivered code.
- No placeholder / stub implementations — every function must be runnable.
- No unrequested refactoring of modules outside the current task scope.
- No global `pip install`; use the project `.venv` and `requirements/*.txt`.
- No `torch.cuda` paths — this project targets MPS (Apple Silicon) with CPU fallback. CUDA support is via the OS-specific `requirements/` files only.
- No raw dataset downloads outside `src/data/partition.py`'s `download_and_tokenize()`.

## Config & Experiments

- All sweep parameters live in `configs/config.yaml`. CLI args override for single-run only.
- Seeds must be explicit and propagated (`config["training"]["seed"]`).
- Results go to `results/`; notebooks go to `notebooks/`. Neither is committed to git (except `.gitkeep` and summary files).

## Git Hygiene

- Commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:` prefixes.
- Never commit model weights, datasets, or `.pth`/`.safetensors` files.

## Workflow

- Read `MEMORY.md` and `PROGRESS.md` at session start — do not re-scan the full repo.
- Work only on the `[ ] Current` task in `PROGRESS.md`.
- After completing a task: update `PROGRESS.md` (check off, set next), append settled decisions to `MEMORY.md`.
