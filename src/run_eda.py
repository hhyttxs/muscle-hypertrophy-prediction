"""Reproduce participant cleaning and descriptive EDA from Mobley 2018 S2.

Run from any directory: python path/to/src/run_eda.py
The raw XLSX stays unchanged. This script does not fit prediction models.
"""
from pathlib import Path
import hashlib
import json
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/mobley_2018_s2_raw.xlsx"
EXPECTED_SHA256 = "efe5f3b5fdfea6a9125fd8ccf73f151af1c91b2ec93d14b05c84d26ad76468ec"
SHEET = "Cluster from VL thickness"
SOURCE_URL = "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0195203.s003&type=supplementary"
PAPER_URL = "https://doi.org/10.1371/journal.pone.0195203"
PARENT_URL = "https://doi.org/10.3390/nu9090972"

# name, original column, unit, timing, analytical role
VARIABLES = [
    ("participant_id", "subject", "identifier", "constant", "join key"),
    ("supplement_code", "protein_group", "category", "assigned group", "sensitivity adjustment; labels unresolved"),
    ("age_years", "Age", "years", "PRE", "sensitivity predictor"),
    ("pre_vl_cm", "T1_VL", "cm", "PRE", "primary baseline predictor"),
    ("post_vl_cm", "T39_VL", "cm", "POST", "primary outcome"),
    ("pre_lean_mass_kg", "T1_LBM_kg", "kg", "PRE", "primary baseline predictor"),
    ("post_lean_mass_kg", "T39_LBM", "kg", "POST", "secondary outcome"),
    ("pre_fat_mass_kg", "T1_FatMass_kg", "kg", "PRE", "primary baseline predictor"),
    ("post_fat_mass_kg", "T39_FatMass", "kg", "POST", "descriptive only"),
    ("pre_squat_3rm_kg", "T1_3RMsquat_kg", "kg", "PRE", "secondary predictor; 3 missing"),
    ("post_squat_3rm_kg", "T39_3RMsquat_kg", "kg", "POST", "secondary outcome; 3 missing"),
    ("pre_energy_kcal_day", "T1_kcal_d", "kcal/day", "PRE 4-day food log", "primary baseline predictor"),
    ("post_energy_kcal_day", "T39_kcal_d", "kcal/day", "POST 4-day food log", "process association only"),
    ("pre_protein_g_day", "T1_PRO_d", "g/day", "PRE 4-day food log", "primary baseline predictor"),
    ("post_protein_g_day", "T39_PRO_d", "g/day", "POST 4-day food log", "process association only; supplement accounting unresolved"),
    ("pre_carbohydrate_g_day", "T1_CHO_d", "g/day", "PRE 4-day food log", "descriptive only"),
    ("post_carbohydrate_g_day", "T39_CHO_d", "g/day", "POST 4-day food log", "descriptive only"),
    ("pre_fat_g_day", "T1_FAT_d", "g/day", "PRE 4-day food log", "descriptive only"),
    ("post_fat_g_day", "T39_FAT_d", "g/day", "POST 4-day food log", "descriptive only"),
    ("training_volume_source_units", "Volume_Lifted", "source units; physical unit not independently verified", "whole 12-week intervention", "process association only"),
]
DERIVED = [
    ("delta_vl_cm", "post_vl_cm - pre_vl_cm", "cm", "POST minus PRE", "descriptive change; never a predictor"),
    ("delta_vl_pct", "100 * delta_vl_cm / pre_vl_cm", "%", "POST minus PRE", "secondary descriptive change; baseline denominator coupling"),
    ("delta_lean_mass_kg", "post_lean_mass_kg - pre_lean_mass_kg", "kg", "POST minus PRE", "secondary descriptive outcome"),
    ("delta_fat_mass_kg", "post_fat_mass_kg - pre_fat_mass_kg", "kg", "POST minus PRE", "descriptive outcome"),
    ("delta_squat_3rm_kg", "post_squat_3rm_kg - pre_squat_3rm_kg", "kg", "POST minus PRE", "secondary descriptive outcome; 3 missing"),
]
CORE = ["pre_vl_cm", "post_vl_cm", "pre_lean_mass_kg", "pre_fat_mass_kg", "pre_protein_g_day", "pre_energy_kcal_day"]


def clean_data():
    checksum = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError("Raw source changed: review the new file before running this protocol.")
    raw = pd.read_excel(RAW, sheet_name=SHEET, engine="openpyxl")
    mask = raw["subject"].astype("string").str.fullmatch(r"MPS\d+", na=False)
    people = raw.loc[mask].copy()
    people.insert(0, "source_excel_row", people.index + 2)
    people = people.sort_values("subject").reset_index(drop=True)
    if len(people) != 67 or people["subject"].nunique() != 67:
        raise ValueError("Expected 67 unique participants; source layout or selection changed.")
    for c in people.columns:
        if c not in ("subject", "source_excel_row"):
            people[c] = pd.to_numeric(people[c], errors="raise")

    clean = pd.DataFrame({"source_excel_row": people["source_excel_row"]})
    for name, original, *_ in VARIABLES:
        clean[name] = people[original]
    clean["supplement_code"] = clean["supplement_code"].astype("Int64").astype("string")
    clean["delta_vl_cm"] = clean["post_vl_cm"] - clean["pre_vl_cm"]
    clean["delta_vl_pct"] = 100 * clean["delta_vl_cm"] / clean["pre_vl_cm"]
    clean["delta_lean_mass_kg"] = clean["post_lean_mass_kg"] - clean["pre_lean_mass_kg"]
    clean["delta_fat_mass_kg"] = clean["post_fat_mass_kg"] - clean["pre_fat_mass_kg"]
    clean["delta_squat_3rm_kg"] = clean["post_squat_3rm_kg"] - clean["pre_squat_3rm_kg"]
    assert clean[CORE].notna().all().all(), "Primary fields must be complete."
    assert (clean[["pre_vl_cm", "post_vl_cm"]] > 0).all().all()
    assert np.allclose(clean["delta_vl_cm"], people["VL Thickness"], atol=1e-10, rtol=0)
    assert np.isclose(clean["delta_vl_cm"].mean(), statistics.fmean(clean["delta_vl_cm"]))
    counts = people.groupby("deltaVL_response_cluster").size().to_dict()
    assert counts == {1: 17, 2: 29, 3: 21}, "Published cluster counts no longer match."

    dictionary = []
    dictionary.append(dict(variable="source_excel_row", original="Excel row number", unit="row", timing="source metadata", role="audit trail", nonmissing_n=67))
    for name, original, unit, timing, role in VARIABLES + DERIVED:
        dictionary.append(dict(variable=name, original=original, unit=unit, timing=timing, role=role,
                               nonmissing_n=int(clean[name].notna().sum())))
    pd.DataFrame(dictionary).to_csv(ROOT / "data/processed/core_data_dictionary.csv", index=False, lineterminator="\n")
    people.to_csv(ROOT / "data/processed/participants_all_source_columns.csv", index=False, lineterminator="\n")
    clean.to_csv(ROOT / "data/processed/participants_core.csv", index=False, lineterminator="\n")
    pd.DataFrame([dict(original_column=c, nonmissing_n=int(people[c].notna().sum()),
                       missing_n=int(people[c].isna().sum())) for c in people.columns]).to_csv(
                           ROOT / "data/processed/source_column_missingness.csv", index=False, lineterminator="\n")
    manifest = {
        "source_doi": "10.1371/journal.pone.0195203", "source_url": SOURCE_URL,
        "license": "CC BY 4.0 (original authors)", "retrieval_date": "2026-10-11",
        "sha256": checksum, "bytes": RAW.stat().st_size, "sheet": SHEET,
        "source_dimensions": "A1:DD96", "source_header_columns": 108,
        "participant_n": 67, "selection_rule": "subject fullmatches MPS[0-9]+",
        "removed_nonparticipant_nonempty_rows": int(raw.loc[~mask].dropna(how="all").shape[0]),
        "excluded_row_types": ["Mean", "StdDev", "Count", "t-test summaries", "legend"],
        "cleaning": ["retain original Excel row", "sort by subject", "preserve missing as empty CSV cells",
                     "no imputation", "no outlier deletion", "no sample-wide transformations"],
        "main_field_complete_n": int(clean[CORE].notna().all(axis=1).sum()),
        "squat_missing_ids": clean.loc[clean.pre_squat_3rm_kg.isna(), "participant_id"].tolist(),
        "supplement_code_counts": {str(k): int(v) for k, v in clean.supplement_code.value_counts().sort_index().items()},
        "outcome_delta_max_difference_from_author_column": float(np.max(np.abs(clean.delta_vl_cm - people["VL Thickness"]))),
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                    "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }
    (ROOT / "data/source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return clean, manifest


def describe(clean):
    records = []
    for name in clean.select_dtypes(include="number").columns:
        if name == "source_excel_row":
            continue
        v = clean[name].dropna().to_numpy(dtype=float)
        records.append(dict(variable=name, n=len(v), missing_n=int(clean[name].isna().sum()),
                            mean=float(v.mean()), sd=float(v.std(ddof=1)), median=float(np.median(v)),
                            q1=float(np.quantile(v, .25)), q3=float(np.quantile(v, .75)),
                            minimum=float(v.min()), maximum=float(v.max())))
    summary = pd.DataFrame(records)
    summary.to_csv(ROOT / "results/descriptive_summary.csv", index=False, lineterminator="\n")
    delta = clean.delta_vl_cm.to_numpy()
    sem = delta.std(ddof=1) / np.sqrt(len(delta))
    margin = t.ppf(.975, len(delta) - 1) * sem
    results = dict(
        n=len(clean), mean_pre_vl_cm=float(clean.pre_vl_cm.mean()),
        mean_post_vl_cm=float(clean.post_vl_cm.mean()), mean_delta_vl_cm=float(delta.mean()),
        sd_delta_vl_cm=float(delta.std(ddof=1)), median_delta_vl_cm=float(np.median(delta)),
        min_delta_vl_cm=float(delta.min()), max_delta_vl_cm=float(delta.max()),
        positive_observed_change_n=int((delta > 1e-10).sum()),
        negative_observed_change_n=int((delta < -1e-10).sum()),
        zero_observed_change_n=int((np.abs(delta) <= 1e-10).sum()),
        mean_delta_vl_cm_t95_interval=[float(delta.mean() - margin), float(delta.mean() + margin)],
        mean_delta_lean_mass_kg=float(clean.delta_lean_mass_kg.mean()),
        mean_pre_protein_g_day=float(clean.pre_protein_g_day.mean()),
        mean_pre_energy_kcal_day=float(clean.pre_energy_kcal_day.mean()),
    )
    (ROOT / "results/eda_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, summary.set_index("variable")


def make_figures(clean, results):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    navy, teal, red = "#17324D", "#168B8A", "#C65353"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.55]})
    ax = axes[0]
    for pre, post in zip(clean.pre_vl_cm, clean.post_vl_cm):
        ax.plot([0, 1], [pre, post], color=teal if post >= pre else red, alpha=.22, linewidth=1)
    ax.scatter(np.zeros(len(clean)), clean.pre_vl_cm, s=12, color=navy, alpha=.6)
    ax.scatter(np.ones(len(clean)), clean.post_vl_cm, s=12, color=teal, alpha=.6)
    ax.plot([0, 1], [results["mean_pre_vl_cm"], results["mean_post_vl_cm"]], color=navy,
            linewidth=3.5, marker="o", markersize=7, label="Sample mean")
    ax.set(xticks=[0, 1], xticklabels=["Before training", "After 12 weeks"],
           ylabel="Vastus lateralis thickness (cm)", title="Paired muscle measurements")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-.18, 1.18)
    ax.grid(axis="y", alpha=.15)
    ax = axes[1]
    changes = np.sort(clean.delta_vl_cm.to_numpy())
    x = np.arange(1, len(changes) + 1)
    ax.vlines(x, 0, changes, color=np.where(changes < 0, red, teal), alpha=.75, linewidth=2)
    ax.scatter(x, changes, s=13, color=np.where(changes < 0, red, teal), zorder=3)
    ax.axhline(0, color=navy, linewidth=.9)
    ax.axhline(results["mean_delta_vl_cm"], color=navy, linestyle="--", linewidth=1.2,
               label=f'Mean observed change: {results["mean_delta_vl_cm"]:.3f} cm')
    ax.set(xlabel="Participant rank by observed change (n = 67)", ylabel="POST - PRE thickness (cm)",
           title="Observed changes vary across participants", xlim=(0, 68))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.15)
    fig.suptitle("Muscle thickness before and after 12 weeks of resistance training", fontsize=15, fontweight="bold", y=.99)
    fig.text(.01, .015, "Source: Mobley et al. (2018), S2 raw data. Changes include measurement variation; ranks are not biological responder classes.", fontsize=8, color="#465566")
    fig.tight_layout(rect=[0, .055, 1, .94])
    fig.savefig(ROOT / "figures/01_muscle_thickness.png", dpi=200)
    fig.savefig(ROOT / "figures/01_muscle_thickness.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, label in zip(axes, ["lean_mass_kg", "fat_mass_kg"], ["DXA lean body mass (kg)", "DXA fat mass (kg)"]):
        for pre, post in zip(clean["pre_" + metric], clean["post_" + metric]):
            ax.plot([0, 1], [pre, post], color=teal, alpha=.23, linewidth=1)
        ax.plot([0, 1], [clean["pre_" + metric].mean(), clean["post_" + metric].mean()],
                color=navy, marker="o", linewidth=3, label="Sample mean")
        ax.set(xticks=[0, 1], xticklabels=["PRE", "POST"], ylabel=label, title=label)
        ax.set_xlim(-.15, 1.15)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False)
    fig.suptitle("Body composition measurements (67 paired records)", fontsize=14, fontweight="bold")
    fig.text(.01, .015, "Source: Mobley et al. (2018), S2. DXA lean body mass is not a direct measure of skeletal muscle mass.", fontsize=8)
    fig.tight_layout(rect=[0, .055, 1, .94])
    fig.savefig(ROOT / "figures/02_body_composition.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, name, unit in zip(axes, ["pre_protein_g_day", "pre_energy_kcal_day"], ["Protein (g/day)", "Energy (kcal/day)"]):
        ax.hist(clean[name], bins=10, color=teal, edgecolor="white", alpha=.88)
        ax.axvline(clean[name].median(), color=navy, linestyle="--", label=f'Median: {clean[name].median():.1f}')
        ax.set(xlabel=unit, ylabel="Participants", title="Baseline " + unit.split(" (")[0].lower())
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Baseline self-reported nutrition (4-day food logs)", fontsize=14, fontweight="bold")
    fig.text(.01, .015, "Source: Mobley et al. (2018), S2. No causal or predictive relationship with muscle growth is shown in these distributions.", fontsize=8)
    fig.tight_layout(rect=[0, .055, 1, .94])
    fig.savefig(ROOT / "figures/03_baseline_nutrition.png", dpi=200)
    plt.close(fig)


def write_report(results, summary, manifest):
    rows = [
        ("训练前股外侧肌厚度", "pre_vl_cm", "cm"), ("12 周后股外侧肌厚度", "post_vl_cm", "cm"),
        ("肌肉厚度变化", "delta_vl_cm", "cm"), ("基线蛋白质摄入", "pre_protein_g_day", "g/day"),
        ("基线热量摄入", "pre_energy_kcal_day", "kcal/day"), ("基线 DXA 瘦体重", "pre_lean_mass_kg", "kg"),
        ("基线脂肪质量", "pre_fat_mass_kg", "kg"), ("基线深蹲 3RM", "pre_squat_3rm_kg", "kg"),
    ]
    lines = ["# 增肌项目：清洗与初步描述性分析", "", "阶段：数据准备与描述性分析。尚未拟合预测模型或检验营养与增肌的关联。", "",
             "## 研究问题", "", "训练前的身体组成、蛋白质和热量摄入，能否在初始肌肉厚度之外，帮助解释和预测 12 周后的肌肉厚度？", "",
             "## 数据清洗", "", "从原始 Excel 中仅保留 subject 匹配 MPS 加数字的参与者行。排除作者的均值、标准差、计数、检验汇总和图例。保留原始行号和全部原始列，另生成清晰命名的核心数据表。", "",
             "- 67 名独立参与者，每人一行。", "- 六个主要分析字段均完整，主要分析保留 67 人。",
             "- 深蹲力量缺失 3 人，不因这一项删除主要分析样本。", "- 未插补、未剔除极值、未将空值填成 0。",
             "- 所有人的厚度差值均与作者已有的 VL Thickness 列一致。", "",
             "## 实际描述性结果", "", "| 指标 | 有效人数 | 均值 ± 标准差 | 中位数 | 最小值至最大值 |", "|---|---:|---|---:|---|"]
    for title, variable, unit in rows:
        s = summary.loc[variable]
        lines.append(f'| {title} ({unit}) | {int(s["n"])} | {s["mean"]:.3f} ± {s["sd"]:.3f} | {s["median"]:.3f} | {s["minimum"]:.3f} 至 {s["maximum"]:.3f} |')
    lo, hi = results["mean_delta_vl_cm_t95_interval"]
    lines += ["", f'观察到的肌肉厚度平均变化为 {results["mean_delta_vl_cm"]:.3f} cm；中位数为 {results["median_delta_vl_cm"]:.3f} cm；范围为 {results["min_delta_vl_cm"]:.2f} 至 {results["max_delta_vl_cm"]:.2f} cm。',
              f'其中 {results["positive_observed_change_n"]} 人测量值增加，{results["negative_observed_change_n"]} 人减少，{results["zero_observed_change_n"]} 人未变。不能把减少者直接定义为“无反应者”或“训练无效”。',
              f'平均变化的常规 t 法 95% 区间为 {lo:.3f} 至 {hi:.3f} cm。这是样本均值的不确定性区间，不是个人增肌的预测区间，也没有分离测量误差。', "",
              "![肌肉厚度前后变化](../figures/01_muscle_thickness.png)", "", "![身体组成](../figures/02_body_composition.png)", "", "![基线营养](../figures/03_baseline_nutrition.png)", "",
              "## 已固定的主要分析范围", "", "- 主要结局：POST 肌肉厚度（cm）；PRE 肌肉厚度作为基线调整项。",
              "- 四个新增主要特征：PRE DXA 瘦体重、PRE 脂肪质量、PRE 蛋白质 g/day、PRE 能量 kcal/day。",
              "- 先与只有 PRE 厚度的模型比较，再评估四个新增特征是否带来样本外改善。",
              "- 年龄和 supplement_code 分类调整列为预先声明的敏感性分析。深蹲力量为次要探索，不作为主要样本筛选条件。",
              "- POST 营养、累计训练量、POST 生物标志物及作者的增肌聚类标签不进入训练前预测。",
              "- 尚未计算这些特征的关联结果。不能根据描述性分布宣布蛋白质或热量能预测增肌。", "",
              "## 需要保留的限制", "", "1. 这是一项公开数据二次分析；原论文已经研究增肌反应。本项目可贡献可复现处理、明确的营养/身体组成建模问题及严格验证，不能声称首次发现个体差异。",
              "2. 67 人为选定研究样本，并非一般人群随机抽样；不能外推到女性、老年人或长期训练者。",
              "3. 数据中没有实际体重或身高列，因此暂不计算蛋白质 g/kg 体重、BMI 或身高标准化瘦体重。DXA 瘦体重不等于纯肌肉质量。",
              "4. 原表 T1=PRE，T39=POST。母研究还收集第 6 周饮食，但本附件没有 T20 饮食列；不能伪造中期记录。",
              "5. 营养是时间点前的 4 天自报记录，不能代表整个 12 周摄入。当前查阅材料未明确确认原始 POST 营养数值是否计入补充剂，保持为待核对问题。",
              "6. 补充剂代码为 4、5、6、7、9，数字到补充剂名称的映射未明确；保留为类别代码，不猜测名称、不当作连续数值。",
              "7. 所有组均训练，本附件没有不训练对照组，也没有重复独立训练周期。观察到的变化差异包含真实差异和测量/日常波动，不能直接视为稳定的先天增肌能力。",
              "8. PRE 与 POST-PRE 的相关性存在数学耦合和回归均值问题；主要模型用 POST 结局调整 PRE，变化和百分比变化只作补充解释。",
              "9. 原始部分缺失原因以单元格颜色和图例表示。源文件保留，但当前导出没有逐格解释颜色；核心字段无缺失，分子变量不进入当前主分析。", "",
              "## 下一步", "", "按照 docs/analysis_protocol.md 拟合基线模型和扩展模型，报告系数及不确定性，开展参与者层面的重复交叉验证；所有预处理和调参只在训练折内完成。负的样本外 R²或未改善的结果也如实记录。", "",
              "## 来源", "", f'1. Mobley et al. (2018), PLOS ONE. {PAPER_URL}', f'2. S2 原始数据：{SOURCE_URL}',
              f'3. 同一母研究的研究方法：Mobley et al. (2017), Nutrients. {PARENT_URL}',
              "4. 原始数据作者使用 CC BY 4.0；本次清洗和描述性分析属于衍生处理，保留原始作者署名与引用。", ""]
    (ROOT / "results/Step1_EDA_Report_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    for relative in ["data/processed", "results", "figures"]:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    clean, manifest = clean_data()
    results, summary = describe(clean)
    make_figures(clean, results)
    write_report(results, summary, manifest)
    print(json.dumps(results, indent=2))
    print("Complete: cleaned participant data, dictionaries, descriptive summaries and three figures.")


if __name__ == "__main__":
    main()
