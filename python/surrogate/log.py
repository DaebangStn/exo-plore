from python.surrogate.util import get_current_version, PROJECT_ROOT
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any


class MemoLogger:
    """Simple TensorBoard logger without Lightning dependency."""

    def __init__(self, name: str, model_type: str = "nn", memo: Optional[str] = None, **kwargs):
        self.exp_name = name
        self._model_type = model_type
        self._memo = memo
        self._count = get_current_version(name) + 1

        version_str = f"v{self._count:02d}" + (f"_{self._memo}" if self._memo else "")
        self._log_dir = PROJECT_ROOT / "logs" / name / version_str
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(log_dir=str(self._log_dir))
        print(f"Logging to {self._log_dir}")

    @property
    def log_dir(self) -> str:
        return str(self._log_dir)

    @property
    def version(self) -> str:
        return f"v{self._count:02d}" + (f"_{self._memo}" if self._memo else "")

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def close(self):
        """Close the TensorBoard writer."""
        self._writer.close()
