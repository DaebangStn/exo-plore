# Comparison between the model prediction and the ground truth
from python.surrogate.preprocess import gt_from_exp_name, gt_stat_from_exp_name
from python.plot.util import *
from tqdm import tqdm


CKPTs = {
    'With\nRectus femoris\n(Proposed)': [
'no_exo_meta_ma15_10000_0801_111350+memo_b20_ma15_10000_0801_111350_pf_param_no_mod_state_ma15_gems+_on_0826_224549',
'no_exo_meta_ma15_10000_0801_111421+memo_b20_ma15_10000_0801_111421_pf_param_no_mod_state_ma15_gems+_on_0826_224557',
'no_exo_meta_ma15_10000_0801_205540+memo_b20_ma15_10000_0801_205540_pf_param_no_mod_state_ma15_gems+_on_0826_224606',
'no_exo_meta_ma15_10000_0802_064456+memo_b20_ma15_10000_0802_064456_pf_param_no_mod_state_ma15_gems+_on_0826_224615',
'no_exo_meta_ma15_10000_0802_163808+memo_b20_ma15_10000_0802_163808_pf_param_no_mod_state_ma15_gems+_on_0826_224624',
'no_exo_meta_ma15_10000_0805_215352+memo_b20_ma15_10000_0805_215352_pf_param_no_mod_state_ma15_gems+_on_0826_224633',
'no_exo_meta_ma15_10000_0805_215354+memo_b20_ma15_10000_0805_215354_pf_param_no_mod_state_ma15_gems+_on_0826_224641',
'no_exo_meta_ma15_10000_0805_215359+memo_b20_ma15_10000_0805_215359_pf_param_no_mod_state_ma15_gems+_on_0826_224649',
'no_exo_meta_ma15_10000_0806_131413+memo_b20_ma15_10000_0806_131413_pf_param_no_mod_state_ma15_gems+_on_0826_224658',
'no_exo_meta_ma15_10000_0806_131416+memo_b20_ma15_10000_0806_131416_pf_param_no_mod_state_ma15_gems+_on_0826_224707',
'no_exo_meta_ma15_10000_0806_131419+memo_b20_ma15_10000_0806_131419_pf_param_no_mod_state_ma15_gems+_on_0826_224716',
    ],

    'Deactivated\nRectus femoris': [
'no_recfem_10000_0906_100709+memo_b21_rf_vas_10000_0906_100709_pf_param_no_mod_state_ma15+_on_0908_184457',
'no_recfem_10000_0906_100713+memo_b21_rf_vas_10000_0906_100713_pf_param_no_mod_state_ma15+_on_0908_184505',
'no_recfem_10000_0906_171934+memo_b21_rf_vas_10000_0906_171934_pf_param_no_mod_state_ma15+_on_0908_184514',
'no_recfem_10000_0906_203023+memo_b21_rf_vas_10000_0906_203023_pf_param_no_mod_state_ma15+_on_0908_184522',
'no_recfem_10000_0907_055743+memo_b21_rf_vas_10000_0907_055743_pf_param_no_mod_state_ma15+_on_0908_184530',
'no_recfem_10000_0907_152836+memo_b21_rf_vas_10000_0907_152836_pf_param_no_mod_state_ma15+_on_0908_132722',
'no_recfem_10000_0908_012340+memo_b21_rf_vas_10000_0908_012340_pf_param_no_mod_state_ma15+_on_0908_132731',
'no_recfem_10000_0909_071246+memo_b21_rf_vas_10000_0909_071246_pf_param_no_mod_state_ma15+_on_0916_173052',
'no_recfem_10000_0909_170200+memo_b21_rf_vas_10000_0909_170200_pf_param_no_mod_state_ma15+_on_0916_173100',
'no_recfem_10000_0910_030109+memo_b21_rf_vas_10000_0910_030109_pf_param_no_mod_state_ma15+_on_0916_173108',
'no_recfem_10000_0921_120254+memo_b21_rf_vas_10000_0921_120254_pf_param_no_mod_state_ma15_gems+_on_0922_134809',
    ],
}

INPUT_COL = ['Phase', 'Stride']
OUTPUT_COL = ['cot_ma15']
transform = 'metabolic'
YRANGE = [0.8, 1.21]
axis_font = 14

def compute_cot(ckpt: str):
    data = gt_from_exp_name(ckpt, INPUT_COL, OUTPUT_COL, transform, is_ckpt=False)[0].collect()
    gt_stat = gt_stat_from_exp_name(ckpt, INPUT_COL, OUTPUT_COL, transform, is_ckpt=False)
    data = denormalize(data, gt_stat)
    return data[OUTPUT_COL].mean()



def main():
    import matplotlib.pyplot as plt

    # Aggregate per-category CoT mean and std by averaging across ckpts in each category
    categories = list(CKPTs.keys())
    category_means = []
    category_mean_stds = []
    category_ns = []
    base_cot = None

    for label in tqdm(categories, desc="Computing CoT stats per category"):
        ckpt_list = CKPTs[label]
        per_ckpt_means = []
        for ckpt in ckpt_list:
            try:
                mean_cot = compute_cot(ckpt)
                # Extract scalar value from polars DataFrame result
                if hasattr(mean_cot, 'item'):
                    per_ckpt_means.append(float(mean_cot.item()))
                else:
                    per_ckpt_means.append(float(mean_cot[0]))
            except Exception as e:
                print(f"Failed computing CoT stats for {ckpt}: {e}")
                continue
        # Aggregate mean CoT across ckpts in this category
        mean_cot = np.nanmean(per_ckpt_means) if len(per_ckpt_means) else np.nan
        if base_cot is None:
            base_cot = mean_cot
        category_means.append(mean_cot / base_cot)
        std_cot = np.nanstd(per_ckpt_means) if len(per_ckpt_means) else np.nan
        category_mean_stds.append(std_cot / base_cot)
        n_valid = int(np.sum(~np.isnan(np.array(per_ckpt_means, dtype=float))))
        category_ns.append(n_valid)

    # Plot single bar chart: Mean CoT per category (bars labeled by dict keys)
    x = np.arange(len(categories))
    width = 0.6
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.2), 5))
    fig.subplots_adjust(
        left=0.15,
        right=0.96, 
        top=0.92,
        bottom=0.20,
    )
    bars = ax.bar(x, category_means, width, yerr=category_mean_stds, capsize=4, label='CoT')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title('Normalized Cost of Transport', fontsize=FONT_SIZE_LABEL)
    # ax.set_title(r'Cost of Transport ($\pm$ SD)', fontsize=FONT_SIZE_LABEL)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_ylim(YRANGE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)

    # Annotate mean value and n on top of each bar
    for i, b in enumerate(bars):
        height = b.get_height()
        if not np.isnan(height):
            # Show mean value on top of bar
            ax.annotate(f"CoT={height:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(-5, 8),
                        textcoords="offset points",
                        ha='right', va='bottom', fontsize=FONT_SIZE_LABEL-4)
            
            # Show n below the mean value
            ax.annotate(f"n={category_ns[i]}",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(-5, 30),
                        textcoords="offset points",
                        ha='right', va='bottom', fontsize=FONT_SIZE_LABEL-4)

    plt.savefig(PLOT_SAVE_DIR / "supp_l8.png", dpi=100, bbox_inches='tight', transparent=True)


if __name__ == "__main__":
    main()
