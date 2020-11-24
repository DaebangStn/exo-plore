# Import compat first to register legacy module aliases for old checkpoints
import python.gait_generator.compat  # noqa: F401
from python.gait_generator.model import *


CKPT = 'ray_results/4800_exo'
OUTDIR = 'ray_results/ts'
EPOCH = None
METADATA = None
# METADATA = 'ray_results/checkpoint-4800/metadata.txt'
PRINT_LOG = True

# When running this script standalone, export models to TorchScript
def cvt_ckpt_to_ts(ckpt_path: str, out_dir: str = OUTDIR, metadata_path: Optional[str] = METADATA,
                   epoch: Optional[int] = EPOCH, override_ckpt_name: Optional[str] = None) -> Path:
    if PRINT_LOG:
        print(f"[CVT_CKPT_TO_TS] ckpt_path: {ckpt_path}, out_dir: {out_dir}, metadata_path: {metadata_path}, epoch: {epoch}, override_ckpt_name: {override_ckpt_name}")
    assert Path(ckpt_path).exists(), f"Checkpoint {ckpt_path} does not exist."
    assert metadata_path is None or Path(metadata_path).is_file(), f"Metadata file {metadata_path} does not exist."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if override_ckpt_name:
        name = override_ckpt_name
    else:
        name = Path(ckpt_path).name
    if PRINT_LOG:
        print("name : ", name)
    if Path(ckpt_path).is_dir():
        ray_outs = [d for d in Path(ckpt_path).iterdir() if d.is_dir()]
        if len(ray_outs) == 0:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
        elif ray_outs[0].name.startswith("checkpoint"):
            if PRINT_LOG:
                print("Checkpoint directory is given")
            latest_outs = Path(ckpt_path)
        else:
            if PRINT_LOG:
                print("Using the latest checkpoint")
            latest_outs = sorted(ray_outs, key=lambda d: d.name, reverse=True)[0]

        latest_outs = Path(latest_outs)
        if epoch:
            ckpt_path = list(latest_outs.glob(f"checkpoint_{int(epoch):06d}/checkpoint-{epoch}*"))
        else:
            ckpt_path = list(latest_outs.glob("max_checkpoint*"))

        if len(ckpt_path) == 0:
            raise FileNotFoundError(f"No checkpoint found in {latest_outs}")
        elif len(ckpt_path) > 1:
            print(f"[WARNING] {len(ckpt_path)} checkpoints found in {latest_outs}")
        ckpt_path = str(ckpt_path[0])
        if PRINT_LOG:
            print("Checkpoint from file ", ckpt_path)
            print("Epoch: ", epoch if epoch else "max_checkpoint")
    else:
        if PRINT_LOG:
            print("Checkpoint from file ", ckpt_path)

    metadata = metadata_from_checkpoint(ckpt_path)
    if metadata_path:
        if PRINT_LOG:
            print("Replacing ckpt's metadata from: ", metadata_path)
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

    output_path = Path(out_dir) / name
    output_path.mkdir(exist_ok=True)

    try:
        if isinstance(metadata, dict):
            metadata_path = output_path / "metadata.yaml"
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f)
        else:
            metadata_path = output_path / "metadata.txt"
            with open(metadata_path, "w") as f:
                f.write(metadata)
    except Exception as e:
        print(f"Failed to write metadata to {metadata_path}: {e}")
    sim_nn, muscle_nn = load_from_checkpoint(ckpt_path, str(metadata_path))
    sim_nn = torch.jit.script(sim_nn)
    torch.jit.save(sim_nn, output_path / "sim_nn.pt")
    if muscle_nn is not None:
        muscle_nn = torch.jit.script(muscle_nn)
        torch.jit.save(muscle_nn, output_path / "muscle_nn.pt")
    if PRINT_LOG:
        print("Torchscript saved to", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to process checkpoints and metadata.")
    parser.add_argument('-c', '--checkpoint', help='checkpoint path', default=CKPT)
    parser.add_argument('-o', '--output', help='output directory', default=OUTDIR)
    parser.add_argument('-m', '--metadata', help='metadata path', default=METADATA)
    parser.add_argument('-e', '--epoch', help='checkpoint epoch', default=EPOCH)
    parser.add_argument('-n', '--name', help='override ckpt name', default=None)
    args = parser.parse_args()
    cvt_ckpt_to_ts(args.checkpoint, args.output, args.metadata, args.epoch, args.name)
