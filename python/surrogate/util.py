from python.plot.util import *

from scipy.optimize import minimize, OptimizeResult, NonlinearConstraint
from sklearn.model_selection import KFold



def debug_compute_graph(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    from torchviz import make_dot
    params = {"x": x, **dict(model.named_parameters())}
    graph = make_dot(y, params=params)
    graph.view()


def get_current_version(exp_name: str) -> int:
    searching_dirs = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    path_with_versions = list(searching_dirs.glob(f"*_v*"))
    matching_dirs = [d for d in path_with_versions if d.is_dir()]
    return len(matching_dirs)



def complete_ckpt_path(ckpt: str, epoch: int = -1) -> Path:
    """
    Complete the checkpoint path for GP models.
    
    Args:
        ckpt: Either full path to a .ckpt file or experiment name
        epoch: Specific epoch number to load (-1 for latest)
    
    Returns:
        Path to the checkpoint file
    """
    # If ckpt is already a full path to a .ckpt file, use it directly
    if ckpt.endswith('.ckpt') and Path(ckpt).exists():
        return Path(ckpt)
    
    ckpt_dir = PROJECT_ROOT / "experiment_data/simulation" / ckpt
    
    # Find all .ckpt files in the directory
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    
    if epoch == -1:
        # Return the latest checkpoint (highest epoch number)
        def extract_epoch(ckpt_path):
            filename = ckpt_path.stem
            # if filename.startswith('epoch_'):
            if filename.startswith('ep'):
                try:
                    return int(filename.split('_')[1])
                except (IndexError, ValueError):
                    return -1
            return -1
        
        ckpt_files = sorted(ckpt_files, key=extract_epoch)
        print(f"Found {len(ckpt_files)} checkpoint files in {ckpt_dir}")
        ckpt_path = ckpt_files[-1]
    else:
        # Find checkpoint with specific epoch - try both naming conventions
        target_filename = f"epoch_{epoch}.ckpt"
        ckpt_path = ckpt_dir / target_filename
        if not ckpt_path.exists():
            # Try the shorter "ep_" format
            target_filename = f"ep_{epoch}.ckpt"
            ckpt_path = ckpt_dir / target_filename
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint file epoch_{epoch}.ckpt or ep_{epoch}.ckpt not found in {ckpt_dir}")
    
    print(f"Using checkpoint: {ckpt_path}")
    return ckpt_path
