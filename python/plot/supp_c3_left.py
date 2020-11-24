from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from python.plot.util import *

EVENTS = {
    r"$\mathrm{w/}\ \mathcal{L}_{\mathrm{IMR}}$": "experiment_data/tfevents/no_exo_meta_ma2",
    r"$\mathrm{wo/}\ \mathcal{L}_{\mathrm{IMR}}$": "experiment_data/tfevents/no_imas",
}

LOSS_TAG = "ray/tune/info/learner/default_policy/custom_metrics/muscle_loss/co_act"
MA_WINDOW = 10
font_size = 16
PLOT_SIZE = 5000

COLORS = [
    '#fde725',
    '#440154',
]

def plot_loss(ax, idx, event_path):
    label = list(EVENTS.keys())[idx]
    event_acc = EventAccumulator(
        event_path, 
        size_guidance={
            "scalars": PLOT_SIZE,
            "images": 0,
            "audio": 0,
            "histograms": 0,
            "tensors": 0,
        }
    )
    event_acc.Reload()
    available_tags = event_acc.Tags()["scalars"]
    assert LOSS_TAG in available_tags, f"LOSS_TAG '{LOSS_TAG}' NOT found in {label}"
    
    loss_events = event_acc.Scalars(LOSS_TAG)
    values = torch.tensor([e.value for e in loss_events])
    mav_values = moving_average(values, MA_WINDOW)
    if "w/" in label:
        print("Multiply by 10")
        mav_values *= 10 
    ax.plot(mav_values, label=label, color=COLORS[idx], linewidth=5)

def main():
    _, ax = plt.subplots(figsize=(4, 8))
    ax.set_xlabel("Training epoch (log)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("\nWithin-group muscle activation difference (log)", fontsize=FONT_SIZE_LABEL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1, PLOT_SIZE])
    ax.set_ylim([1e-4, 3e3])
    ax.set_xticks([1, 10, 100, 1000, 5000])
    ax.set_yticks([1e-4, 1e-2, 1, 1000])
    # ax.tick_params(labelsize=(font_size-4))
    ax.tick_params(axis='both', which='major', direction='in', length=12)
    ax.tick_params(axis='both', which='minor', direction='in', length=4)

     
    tq = tqdm(range(len(EVENTS)), desc="plotting loss", ncols=120)
    for idx in tq:
        event_dir = list(EVENTS.values())[idx]
        event_path = glob(f'{event_dir}/**/events.out.tfevents.*', recursive=True)
        assert len(event_path) == 1, f"found {len(event_path)} event files in {event_dir}"
        event_path = event_path[0]
        tq.set_postfix(event=event_dir)
        plot_loss(ax, idx, event_path)
    
    plt.tight_layout()
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LABEL)
    plt.savefig(PROJECT_ROOT / "plot/supp_c3_left.png", dpi=150, transparent=True)

if __name__ == "__main__":
    main()
