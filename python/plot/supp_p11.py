# Comparison between the model prediction and the ground truth
from python.surrogate.preprocess import gt_from_exp_name, gt_stat_from_exp_name
from python.plot.util import *
from tqdm import tqdm


NUM_WORKERS = 8

CKPTs = {
    # 'normal': [
    #     'no_exo_meta_ma15_10000_0801_111350+memo_b21_U100x10_no_mod_state_ma15+_on_0906_133750',
    #     'no_exo_meta_ma15_10000_0801_111421+memo_b21_U100x10_no_mod_state_ma15+_on_0906_134650',
    #     'no_exo_meta_ma15_10000_0801_205540+memo_b21_U100x10_no_mod_state_ma15+_on_0906_135529',
    # ],
    'Foot drop (severe)': [        
'footdrop_16000_0829_144123+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0906_115227',
'footdrop_16000_0825_165208+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_005713',
'footdrop_16000_0827_001346+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_010601',
'footdrop_16000_0828_035710+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_011432',
'footdrop_16000_0828_223228+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_012331',
'footdrop_16000_0829_144123+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_013155',
'footdrop_16000_0830_091919+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_014051',
'footdrop_16000_0831_104428+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_014956',
'footdrop_16000_0901_095756+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_015903',
'footdrop_16000_0902_095846+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0907_020805',
    ],
    'Foot drop (mild)': [
'footdrop_16000_0829_144123+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0906_120123',
'footdrop_16000_0825_165208+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_021633',
'footdrop_16000_0827_001346+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_022541',
'footdrop_16000_0828_035710+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_023455',
'footdrop_16000_0828_223228+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_024349',
'footdrop_16000_0829_144123+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_025236',
'footdrop_16000_0830_091919+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_030143',
'footdrop_16000_0831_104428+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_031033',
'footdrop_16000_0901_095756+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_031907',
'footdrop_16000_0902_095846+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0907_032802',
    ],

#     'foot-mild2': [
# 'footdrop_16000_0825_165208+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_033630',
# 'footdrop_16000_0827_001346+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_034544',
# 'footdrop_16000_0828_035710+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_035430',
# 'footdrop_16000_0828_223228+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_040338',
# 'footdrop_16000_0829_144123+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_041238',
# 'footdrop_16000_0830_091919+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_042134',
# 'footdrop_16000_0831_104428+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_043034',
# 'footdrop_16000_0901_095756+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_043913',
# 'footdrop_16000_0902_095846+memo_p21_foot5_U100x10_no_mod_state_ma15+_on_0907_044807',
#     ],

#     'Others (severe)': [
# 'calcaneous_16000_0816_120417+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_122530',
# 'calcaneous_16000_0811_134143+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_214928',
# 'calcaneous_16000_0811_142659+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_215811',
# 'calcaneous_16000_0812_035335+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_220703',
# 'calcaneous_16000_0814_094328+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_221604',

# 'equinus_16000_0811_134131+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0906_124253',
# 'equinus_16000_0811_134620+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0906_233454',
# 'equinus_16000_0813_185633+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0906_234929',

# 'crouch_16000_0822_213227+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0906_130111',
# 'crouch_16000_0823_072634+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_045651',
# 'crouch_16000_0824_200136+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_050545',
# 'crouch_16000_0826_100856+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_051438',
# 'crouch_16000_0827_150621+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_052329',

# 'waddling_16000_0817_020413+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0906_131931',
# 'waddling_16000_0811_134550+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0907_064523',
# 'waddling_16000_0811_140715+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0907_065323',
#     ],
    'Calcaneous (severe)': [
'calcaneous_16000_0816_120417+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_122530',
'calcaneous_16000_0811_134143+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_214928',
'calcaneous_16000_0811_142659+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_215811',
'calcaneous_16000_0812_035335+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_220703',
'calcaneous_16000_0814_094328+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_221604',
    ],
    'Equinus (severe)': [
'equinus_16000_0811_134131+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0906_124253',
'equinus_16000_0811_134620+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0906_233454',
'equinus_16000_0813_185633+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0906_234929',
'equinus_16000_0919_135840+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0921_164253',
'equinus_16000_0920_085935+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0921_165145',
    ],
    'Crouch (severe)': [
'crouch_16000_0822_213227+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0906_130111',
'crouch_16000_0823_072634+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_045651',
'crouch_16000_0824_200136+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_050545',
'crouch_16000_0826_100856+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_051438',
'crouch_16000_0827_150621+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0907_052329',
    ],
    'Waddling (severe)': [
'waddling_16000_0817_020413+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0906_131931',
'waddling_16000_0811_134550+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0907_064523',
'waddling_16000_0811_140715+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0907_065323',
'waddling_16000_0913_165602+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0921_172915',
'waddling_16000_0914_055104+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0921_172028',
    ],
    
#     'Others (mild)': [
# 'calcaneous_16000_0816_120417+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_123357',
# 'calcaneous_16000_0811_134143+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_222444',
# 'calcaneous_16000_0811_142659+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_223342',
# 'calcaneous_16000_0812_035335+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_224228',
# 'calcaneous_16000_0814_094328+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_225145',

# 'equinus_16000_0811_134131+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0906_125132',
# 'equinus_16000_0811_134620+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0906_235815',
# 'equinus_16000_0813_185633+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0907_001855',

# 'waddling_16000_0817_020413+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0906_132842',
# 'waddling_16000_0811_134550+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0907_083458',
# 'waddling_16000_0811_140715+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0907_084325',

# 'crouch_16000_0822_213227+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0906_131020',
# 'crouch_16000_0823_072634+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_053227',
# 'crouch_16000_0824_200136+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_054141',
# 'crouch_16000_0826_100856+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_055054',
# 'crouch_16000_0827_150621+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_055956',
#     ],
    'Calcaenous (mild)': [
'calcaneous_16000_0816_120417+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_123357',
'calcaneous_16000_0811_134143+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_222444',
'calcaneous_16000_0811_142659+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_223342',
'calcaneous_16000_0812_035335+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_224228',
'calcaneous_16000_0814_094328+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_225145',
    ],
    'Equinus (mild)': [
'equinus_16000_0811_134131+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0906_125132',
'equinus_16000_0811_134620+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0906_235815',
'equinus_16000_0813_185633+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0907_001855',
'equinus_16000_0919_135840+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0921_170051',
'equinus_16000_0920_085935+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0921_171029',
    ],
    'Crouch (mild)': [
'crouch_16000_0822_213227+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0906_131020',
'crouch_16000_0823_072634+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_053227',
'crouch_16000_0824_200136+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_054141',
'crouch_16000_0826_100856+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_055054',
'crouch_16000_0827_150621+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0907_055956',
    ],
    'Waddling (mild)': [
'waddling_16000_0817_020413+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0906_132842',
'waddling_16000_0811_134550+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0907_083458',
'waddling_16000_0811_140715+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0907_084325',
'waddling_16000_0913_165602+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0921_174746',
'waddling_16000_0914_055104+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0921_173810',
    ],
    
#     'othes-mild2': [
# 'calcaneous_16000_0811_134143+memo_p21_calc4_U100x10_no_mod_state_ma15+_on_0906_230006',
# 'calcaneous_16000_0811_142659+memo_p21_calc4_U100x10_no_mod_state_ma15+_on_0906_230902',
# 'calcaneous_16000_0812_035335+memo_p21_calc4_U100x10_no_mod_state_ma15+_on_0906_231716',
# 'calcaneous_16000_0814_094328+memo_p21_calc4_U100x10_no_mod_state_ma15+_on_0906_232614',

# 'equinus_16000_0811_134620+memo_p21_equi100_U100x10_no_mod_state_ma15+_on_0907_002748',
# 'equinus_16000_0813_185633+memo_p21_equi100_U100x10_no_mod_state_ma15+_on_0907_004824',

# 'crouch_16000_0823_072634+memo_p21_hyp100_U100x10_no_mod_state_ma15+_on_0907_060905',
# 'crouch_16000_0824_200136+memo_p21_hyp100_U100x10_no_mod_state_ma15+_on_0907_061814',
# 'crouch_16000_0826_100856+memo_p21_hyp100_U100x10_no_mod_state_ma15+_on_0907_062716',
# 'crouch_16000_0827_150621+memo_p21_hyp100_U100x10_no_mod_state_ma15+_on_0907_063622',

# 'waddling_16000_0811_134550+memo_p21_wad4_U100x10_no_mod_state_ma15+_on_0907_085227',
# 'waddling_16000_0811_140715+memo_p21_wad4_U100x10_no_mod_state_ma15+_on_0907_090101',
#     ],

    # 'normal': [
    #     'no_exo_meta_ma15_10000_0801_111350+memo_b21_U100x10_no_mod_state_ma15+_on_0906_133750',
    #     'no_exo_meta_ma15_10000_0801_111421+memo_b21_U100x10_no_mod_state_ma15+_on_0906_134650',
    #     'no_exo_meta_ma15_10000_0801_205540+memo_b21_U100x10_no_mod_state_ma15+_on_0906_135529',
    # ],
    # 'foot-severe': [        
    #     'footdrop_16000_0829_144123+memo_b21_foot00_U100x10_no_mod_state_ma15+_on_0906_115227',
    # ],
    # 'foot-moderate': [
    #     'footdrop_16000_0829_144123+memo_b21_foot20_U100x10_no_mod_state_ma15+_on_0906_120123',
    # ],
    # 'calcaneus-severe': [
    #     'calcaneous_16000_0816_120417+memo_b21_calc0_U100x10_no_mod_state_ma15+_on_0906_122530',
    # ],
    # 'calcaneus-moderate': [
    #     'calcaneous_16000_0816_120417+memo_b21_calc2_U100x10_no_mod_state_ma15+_on_0906_123357',
    # ],
    # 'equinus-severe': [
    #     'equinus_16000_0811_134131+memo_b21_equi75_U100x10_no_mod_state_ma15+_on_0906_124253',
    # ],
    # 'equinus-moderate': [
    #     'equinus_16000_0811_134131+memo_b21_equi85_U100x10_no_mod_state_ma15+_on_0906_125132',
    # ],
    # 'hypa-severe': [
    #     'crouch_16000_0822_213227+memo_b21_hyp60_U100x10_no_mod_state_ma15+_on_0906_130111',
    # ],
    # 'hypa-moderate': [
    #     'crouch_16000_0822_213227+memo_b21_hyp80_U100x10_no_mod_state_ma15+_on_0906_131020',
    # ],
    # 'waddiling-severe': [
    #     'waddling_16000_0817_020413+memo_b21_wad2_U100x10_no_mod_state_ma15+_on_0906_132842',
    # ],
    # 'waddiling-moderate': [
    #     'waddling_16000_0817_020413+memo_b21_wad0_U100x10_no_mod_state_ma15+_on_0906_131931',
    # ],
}

INPUT_COL = ['Phase', 'Stride']
OUTPUT_COL = ['cot_ma15']
transform = 'metabolic'
CMAP_THEME = 'viridis'
STD_RANGE = [0, 3]
SHIFT_RANGE = [-2.5, 2.5]
cmap_kwargs = {
    "cmap": CMAP_THEME,
    "vmin": SHIFT_RANGE[0],
    "vmax": SHIFT_RANGE[1],
}
axis_font = 14


def check_failed_exp(min_ratio: float, max_ratio: float, std_ratio: float):
    if std_ratio > 5:
        return True
 

def compute_cv(ckpt: str):
    data = gt_from_exp_name(ckpt, INPUT_COL, OUTPUT_COL, transform, is_ckpt=False)[0].collect()
    gt_stat = gt_stat_from_exp_name(ckpt, INPUT_COL, OUTPUT_COL, transform, is_ckpt=False)
    data = denormalize(data, gt_stat)

    # Compute stats grouped by unique parameter combinations
    grouped_stats = data.group_by(INPUT_COL).agg([
        pl.col(output_col).mean().alias(f"{output_col}_mean") for output_col in OUTPUT_COL
    ] + [
        pl.col(output_col).std().alias(f"{output_col}_std") for output_col in OUTPUT_COL
    ] + [
        pl.len().alias("count")
    ])

    # Drop if any std is null
    std_cols = [f"{field}_std" for field in OUTPUT_COL]
    if any(grouped_stats[col].is_null().any() for col in std_cols):
        grouped_stats = grouped_stats.drop_nulls(std_cols)

    # Compute Coefficient of Variation (CV = std/mean) for each param combo, then mean across combos
    cv_vals = []
    for field in OUTPUT_COL:
        mean_col = f"{field}_mean"
        std_col = f"{field}_std"
        # Avoid division by zero
        cv_series = grouped_stats[std_col] / (grouped_stats[mean_col] + 1e-12)  # type: ignore
        cv_vals.append(cv_series)
    cv_concat = pl.concat(cv_vals, how="horizontal") if len(cv_vals) > 0 else None
    cv_overall = float(np.nanmean(cv_concat.to_numpy())) if cv_concat is not None else float('nan')

    return {"cv": cv_overall}


def main():
    import matplotlib.pyplot as plt

    # Aggregate per-category CV by averaging across ckpts in each category
    categories = list(CKPTs.keys())
    category_cvs = []
    category_cv_stds = []
    category_ns = []

    for label in tqdm(categories, desc="Computing stats per category"):
        ckpt_list = CKPTs[label]
        per_ckpt_cvs = []
        for ckpt in ckpt_list:
            try:
                metrics = compute_cv(ckpt)
                per_ckpt_cvs.append(metrics["cv"] * 100)
            except Exception as e:
                print(f"Failed computing stats for {ckpt}: {e}")
                continue
        # Aggregate CV across ckpts in this category
        category_cvs.append(np.nanmean(per_ckpt_cvs) if len(per_ckpt_cvs) else np.nan)
        category_cv_stds.append(np.nanstd(per_ckpt_cvs) if len(per_ckpt_cvs) else np.nan)
        n_valid = int(np.sum(~np.isnan(np.array(per_ckpt_cvs, dtype=float))))
        category_ns.append(n_valid)

    # Plot single bar chart: CV per category (bars labeled by dict keys)
    x = np.arange(len(categories))
    width = 0.6
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.2), 5))
    fig.subplots_adjust(
        left=0.08,
        right=0.96, 
        top=0.92,
        bottom=0.22,
    )
    bars = ax.bar(x, category_cvs, width, yerr=category_cv_stds, capsize=4, label='CV')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20)
    ax.set_title(r'Coefficient of variation ($\pm$ SD) (%)', fontsize=FONT_SIZE_LABEL)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_ylim(0, 4)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)

    # Annotate n on top of each bar
    for i, b in enumerate(bars):
        height = b.get_height()
        ax.annotate(f"n={category_ns[i]}",
                    xy=(b.get_x() + b.get_width() / 2, 0 if np.isnan(height) else height),
                    xytext=(5, 4),
                    textcoords="offset points",
                    ha='left', va='bottom', fontsize=FONT_SIZE_LABEL)

    plt.savefig(PLOT_SAVE_DIR / "supp_p11.png", dpi=100, bbox_inches='tight', transparent=True)


if __name__ == "__main__":
    main()
