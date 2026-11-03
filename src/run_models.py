"""Stage 2: baseline-adjusted association and leakage-controlled validation.

Run: python src/run_models.py
Uses fixed protocol, retains all participants, and recreates Stage 1 first.
"""
from pathlib import Path
import json
import hashlib
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import run_eda

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/models"
SEED = 20261011
REPEATS = 20
FOLDS = 5
ALPHAS = [0.1, 1.0, 10.0, 100.0]
BASE = ["pre_vl_cm"]
ADDED = ["pre_lean_mass_kg", "pre_fat_mass_kg", "pre_protein_g_day", "pre_energy_kcal_day"]
FULL = BASE + ADDED
UNITS = {"pre_vl_cm": .1, "pre_lean_mass_kg": 10., "pre_fat_mass_kg": 10.,
         "pre_protein_g_day": 25., "pre_energy_kcal_day": 500., "age_years": 1.}
LABELS = {
    "reference_mean_post": "Mean POST reference", "reference_mean_change": "Mean-change reference",
    "baseline_ols": "Initial thickness only", "extended_ols": "Initial + composition + nutrition",
    "extended_ridge": "Same features, tuned ridge", "baseline_age_ols": "Initial + age",
    "extended_age_ols": "Extended + age", "baseline_age_group_ols": "Initial + age + group",
    "extended_age_group_ols": "Extended + age + group",
}
MAIN_MODELS = ["reference_mean_post", "reference_mean_change", "baseline_ols", "extended_ols", "extended_ridge"]
CV_FEATURES = {
    "baseline_ols": (BASE, False), "extended_ols": (FULL, False),
    "baseline_age_ols": (BASE + ["age_years"], False),
    "extended_age_ols": (FULL + ["age_years"], False),
    "baseline_age_group_ols": (BASE + ["age_years"], True),
    "extended_age_group_ols": (FULL + ["age_years"], True),
}


def regression_hc3(X, y):
    """X includes an intercept; return OLS and independently checked HC3."""
    n, p = X.shape
    assert np.linalg.matrix_rank(X) == p
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    bread = np.linalg.inv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, bread, X)
    assert np.all(leverage < 1 - 1e-10)
    correction = resid / (1 - leverage)
    weighted = X * correction[:, None]
    covariance = bread @ (weighted.T @ weighted) @ bread
    covariance = (covariance + covariance.T) / 2
    df = n - p
    se = np.sqrt(np.diag(covariance))
    tvalue = beta / se
    critical = stats.t.ppf(.975, df)
    sse = float(resid @ resid)
    mse_resid = sse / df
    cook = resid**2 * leverage / (p * mse_resid * (1 - leverage)**2)
    studentized_internal = resid / np.sqrt(mse_resid * (1 - leverage))
    # Delete-one fitting is a distinct route to the HC3 covariance:
    # sum_i (beta_without_i - beta)(beta_without_i - beta)' = HC3.
    deleted = np.vstack([np.linalg.lstsq(np.delete(X, i, axis=0), np.delete(y, i), rcond=None)[0]
                         for i in range(n)])
    shifts = deleted - beta
    covariance_delete_one = shifts.T @ shifts
    assert np.allclose(covariance, covariance_delete_one, rtol=1e-7, atol=1e-10), "HC3 verification failed"
    check_fit = LinearRegression(fit_intercept=False).fit(X, y).predict(X)
    assert np.allclose(fitted, check_fit, atol=1e-10), "OLS solver verification failed"
    return dict(beta=beta, covariance=covariance, se=se, tvalue=tvalue,
                pvalue=2 * stats.t.sf(np.abs(tvalue), df),
                ci_low=beta-critical*se, ci_high=beta+critical*se,
                fitted=fitted, residual=resid, leverage=leverage, cook=cook,
                studentized_internal=studentized_internal, deleted_beta=deleted,
                n=n, parameters=p, df_resid=df, r2=float(r2_score(y, fitted)),
                adj_r2=float(1 - (1-r2_score(y, fitted))*(n-1)/df),
                sse=sse, mse_resid=mse_resid,
                hc3_delete_one_max_abs_difference=float(np.max(np.abs(covariance-covariance_delete_one))))


def association_design(d, variables, group=False):
    values = [(d[k].to_numpy(float) - d[k].mean()) / UNITS[k] for k in variables]
    names = ["intercept_at_sample_mean"] + variables
    if group:
        indicators = pd.get_dummies(d["supplement_code"], prefix="supplement_code", drop_first=True, dtype=float)
        values.extend(indicators[k].to_numpy() for k in indicators.columns)
        names.extend(indicators.columns.tolist())
    return np.column_stack([np.ones(len(d))] + values), names


def fit_associations(d):
    y = d.post_vl_cm.to_numpy()
    settings = {
        "baseline_ols": (BASE, False), "extended_ols": (FULL, False),
        "baseline_age_ols": (BASE+["age_years"], False), "extended_age_ols": (FULL+["age_years"], False),
        "baseline_age_group_ols": (BASE+["age_years"], True), "extended_age_group_ols": (FULL+["age_years"], True),
    }
    fits = {}; coefficients = []; summaries = []; diagnostics = []; deleted_records = []
    for model, (variables, group) in settings.items():
        X, names = association_design(d, variables, group)
        fit = regression_hc3(X, y); fits[model] = fit
        for j, name in enumerate(names):
            unit = UNITS.get(name, 1.)
            coefficients.append(dict(model=model, term=name, effect_per_increment=unit,
                                     beta_cm=float(fit["beta"][j]), hc3_se_cm=float(fit["se"][j]),
                                     t_reference_df=fit["df_resid"], t_value=float(fit["tvalue"][j]),
                                     p_value_unadjusted=float(fit["pvalue"][j]),
                                     ci95_low_cm=float(fit["ci_low"][j]), ci95_high_cm=float(fit["ci_high"][j])))
        summary = {k: fit[k] for k in ["n","parameters","df_resid","r2","adj_r2","sse","hc3_delete_one_max_abs_difference"]}
        summary.update(model=model, design_condition_number=float(np.linalg.cond(X)))
        if model.startswith("extended"):
            positions = [names.index(k) for k in ADDED]
            b = fit["beta"][positions]
            c = fit["covariance"][np.ix_(positions,positions)]
            w = float(b @ np.linalg.solve(c,b))
            summary.update(added_block_wald_f_approx=w/len(positions),
                           added_block_wald_p_approx=float(stats.f.sf(w/len(positions),len(positions),fit["df_resid"])))
        summaries.append(summary)
        if model == "extended_ols":
            p=fit["parameters"];n=fit["n"]
            for i, pid in enumerate(d.participant_id):
                diagnostics.append(dict(participant_id=pid, observed_post_cm=float(y[i]),
                    fitted_post_cm=float(fit["fitted"][i]), residual_cm=float(fit["residual"][i]),
                    leverage=float(fit["leverage"][i]), cooks_distance=float(fit["cook"][i]),
                    studentized_residual_internal=float(fit["studentized_internal"][i]),
                    flag_cook_4_over_n=bool(fit["cook"][i]>4/n),flag_leverage_2p_over_n=bool(fit["leverage"][i]>2*p/n)))
                for j,name in enumerate(names):
                    deleted_records.append(dict(omitted_participant_id=pid,term=name,beta_cm=float(fit["deleted_beta"][i,j])))
    coefficient_df=pd.DataFrame(coefficients);summary_df=pd.DataFrame(summaries)
    diagnostic_df=pd.DataFrame(diagnostics)
    coefficient_df.to_csv(RESULTS/"association_coefficients_hc3.csv",index=False, lineterminator="\n")
    summary_df.to_csv(RESULTS/"association_model_summary.csv",index=False, lineterminator="\n")
    diagnostic_df.to_csv(RESULTS/"primary_model_diagnostics.csv",index=False, lineterminator="\n")
    pd.DataFrame(deleted_records).to_csv(RESULTS/"leave_one_participant_out_coefficients.csv",index=False, lineterminator="\n")
    corr=d[FULL].corr();corr.to_csv(RESULTS/"primary_predictor_correlations.csv",lineterminator="\n")
    vif=[]
    for k in FULL:
        other=[x for x in FULL if x!=k]
        pred=LinearRegression().fit(d[other],d[k]).predict(d[other])
        auxiliary_r2=r2_score(d[k],pred)
        vif.append(dict(variable=k,vif=float(1/(1-auxiliary_r2))))
    pd.DataFrame(vif).to_csv(RESULTS/"variance_inflation_factors.csv",index=False, lineterminator="\n")
    return fits,coefficient_df,summary_df,diagnostic_df,pd.DataFrame(vif)


def pipeline_model(variables, group=False):
    transforms=[("numeric",StandardScaler(),variables)]
    if group:
        transforms.append(("group",OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False),["supplement_code"]))
    return Pipeline([("prepare",ColumnTransformer(transforms,remainder="drop")),("model",LinearRegression())])


def metrics(y,pred,pre):
    growth=y-pre;growth_pred=pred-pre
    post_mae=mean_absolute_error(y,pred);post_rmse=np.sqrt(mean_squared_error(y,pred))
    assert np.isclose(post_mae,mean_absolute_error(growth,growth_pred),atol=1e-12)
    assert np.isclose(post_rmse,np.sqrt(mean_squared_error(growth,growth_pred)),atol=1e-12)
    return dict(mae_cm=float(post_mae),rmse_cm=float(post_rmse),
                r2_post=float(r2_score(y,pred)),r2_observed_change=float(r2_score(growth,growth_pred)))


def run_cv(d):
    y=d.post_vl_cm.to_numpy();pre=d.pre_vl_cm.to_numpy();ids=d.participant_id.to_numpy()
    oof_records=[];outer_records=[];inner_records=[];ridge_choices=[];grid_scores=[];fold_scores=[]
    splitter=RepeatedKFold(n_splits=FOLDS,n_repeats=REPEATS,random_state=SEED)
    for ix,(train,test) in enumerate(splitter.split(d)):
        repeat=ix//FOLDS+1;fold=ix%FOLDS+1
        assert not set(train)&set(test)
        assert len(train)+len(test)==len(d)
        for role,indices in [("train",train),("test",test)]:
            outer_records.extend(dict(repeat=repeat,fold=fold,participant_id=ids[j],role=role) for j in indices)
        train_df=d.iloc[train];test_df=d.iloc[test]
        predictions={"reference_mean_post":np.full(len(test),y[train].mean()),
                     "reference_mean_change":pre[test]+(y[train]-pre[train]).mean()}
        for model,(variables,group) in CV_FEATURES.items():
            estimator=pipeline_model(variables,group)
            columns=variables+(["supplement_code"] if group else [])
            estimator.fit(train_df[columns],y[train])
            fitted_scaler=estimator.named_steps["prepare"].named_transformers_["numeric"]
            assert np.allclose(fitted_scaler.mean_,train_df[variables].mean().to_numpy())
            predictions[model]=estimator.predict(test_df[columns])
        inner=KFold(n_splits=5,shuffle=True,random_state=SEED+ix+1)
        inner_splits=list(inner.split(train_df))
        for inner_ix,(it,iv) in enumerate(inner_splits):
            assert not set(train[it])&set(train[iv])
            assert not (set(train[it])|set(train[iv]))&set(test)
            for role,indices in [("train",it),("validation",iv)]:
                inner_records.extend(dict(repeat=repeat,outer_fold=fold,inner_fold=inner_ix+1,
                                          participant_id=ids[train[j]],role=role) for j in indices)
        ridge=Pipeline([("scale",StandardScaler()),("model",Ridge())])
        search=GridSearchCV(ridge,{"model__alpha":ALPHAS},scoring="neg_mean_squared_error",cv=inner_splits,
                            refit=True,n_jobs=1,error_score="raise")
        search.fit(train_df[FULL],y[train])
        assert np.allclose(search.best_estimator_.named_steps["scale"].mean_,train_df[FULL].mean().to_numpy())
        predictions["extended_ridge"]=search.predict(test_df[FULL])
        ridge_choices.append(dict(repeat=repeat,fold=fold,chosen_alpha=float(search.best_params_["model__alpha"]),
                                  inner_mean_mse_cm2=float(-search.best_score_)))
        for j,params in enumerate(search.cv_results_["params"]):
            grid_scores.append(dict(repeat=repeat,fold=fold,alpha=float(params["model__alpha"]),
                                    mean_inner_mse_cm2=float(-search.cv_results_["mean_test_score"][j]),
                                    sd_inner_mse_cm2=float(search.cv_results_["std_test_score"][j]),
                                    rank=int(search.cv_results_["rank_test_score"][j])))
        for model,pred in predictions.items():
            assert np.isfinite(pred).all()
            fold_scores.append(dict(repeat=repeat,fold=fold,model=model,n_test=len(test),**metrics(y[test],pred,pre[test])))
            for j,idx in enumerate(test):
                oof_records.append(dict(repeat=repeat,fold=fold,model=model,participant_id=ids[idx],
                    observed_pre_cm=float(pre[idx]),observed_post_cm=float(y[idx]),predicted_post_cm=float(pred[j]),
                    observed_change_cm=float(y[idx]-pre[idx]),predicted_change_cm=float(pred[j]-pre[idx])))
        if fold==5:print(f"Cross-validation repeat {repeat}/{REPEATS} complete",flush=True)
    oof=pd.DataFrame(oof_records)
    assert (oof.groupby(["repeat","model","participant_id"]).size()==1).all()
    assert (oof.groupby(["repeat","model"]).size()==len(d)).all()
    scores=[]
    for (repeat,model),g in oof.groupby(["repeat","model"],sort=True):
        scores.append(dict(repeat=repeat,model=model,n_test_predictions=len(g),
                           **metrics(g.observed_post_cm.to_numpy(),g.predicted_post_cm.to_numpy(),g.observed_pre_cm.to_numpy())))
    score_df=pd.DataFrame(scores)
    metric_cols=["mae_cm","rmse_cm","r2_post","r2_observed_change"]
    summary=[]
    for model,g in score_df.groupby("model"):
        row=dict(model=model,label=LABELS[model],repeats=len(g))
        for col in metric_cols:
            row.update({col+"_mean":float(g[col].mean()),col+"_sd":float(g[col].std(ddof=1)),
                        col+"_min":float(g[col].min()),col+"_max":float(g[col].max())})
        summary.append(row)
    comparisons=[]
    for extended,baseline in [("extended_ols","baseline_ols"),("extended_ridge","baseline_ols"),
                              ("extended_ols","reference_mean_change"),("extended_ridge","reference_mean_change"),
                              ("extended_age_ols","baseline_age_ols"),("extended_age_group_ols","baseline_age_group_ols")]:
        a=score_df.loc[score_df.model==extended].set_index("repeat")
        b=score_df.loc[score_df.model==baseline].set_index("repeat")
        for repeat in a.index:
            row=dict(repeat=int(repeat),extended=extended,baseline=baseline)
            for col in metric_cols:row[col+"_difference_extended_minus_baseline"]=float(a.loc[repeat,col]-b.loc[repeat,col])
            comparisons.append(row)
    tables={"cv_out_of_fold_predictions":oof,"cv_outer_splits":pd.DataFrame(outer_records),
            "cv_inner_splits":pd.DataFrame(inner_records),"cv_ridge_selection":pd.DataFrame(ridge_choices),
            "cv_ridge_grid_scores":pd.DataFrame(grid_scores),"cv_fold_metrics":pd.DataFrame(fold_scores),
            "cv_repeat_metrics":score_df,"cv_metric_summary":pd.DataFrame(summary),
            "cv_paired_model_differences":pd.DataFrame(comparisons)}
    for name,df in tables.items():df.to_csv(RESULTS/(name+".csv"),index=False, lineterminator="\n")
    return oof,score_df,pd.DataFrame(summary).set_index("model"),pd.DataFrame(comparisons),pd.DataFrame(ridge_choices)


def make_figures(fits,coefficients,diagnostics,summary,score_df):
    navy="#17324D";teal="#168B8A";gold="#BD862B"
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold"})
    models=["reference_mean_change","baseline_ols","extended_ols","extended_ridge"]
    short=["Mean-change\nreference","Initial thickness\nonly","Add composition\n+ nutrition","Same features\nwith ridge"]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    positions=np.arange(len(models))
    means=[summary.loc[m,"mae_cm_mean"] for m in models]
    sds=[summary.loc[m,"mae_cm_sd"] for m in models]
    axes[0].bar(positions,means,color=["#96A6B5",navy,teal,gold],width=.6)
    axes[0].errorbar(positions,means,yerr=sds,fmt="none",ecolor=navy,capsize=4)
    for x,v in zip(positions,means):axes[0].text(x,v+.014,f"{v:.3f}",ha="center",fontsize=11)
    axes[0].set(xticks=positions,xticklabels=short,ylabel="Out-of-fold MAE (cm), lower is better",title="Prediction error on held-out participants")
    axes[0].set_ylim(0,max(means+sds)*1.4)
    for repeat in range(1,REPEATS+1):
        g=score_df[score_df.repeat==repeat].set_index("model")
        axes[1].plot(positions,[g.loc[m,"r2_observed_change"] for m in models],color="#AAB9C5",alpha=.4,linewidth=.7)
    vals=[summary.loc[m,"r2_observed_change_mean"] for m in models]
    axes[1].plot(positions,vals,color=navy,marker="o",linewidth=2.5)
    axes[1].axhline(0,color="#697889",linestyle="--",linewidth=.9)
    axes[1].set(xticks=positions,xticklabels=short,ylabel="R-squared for observed thickness change",title="Predicting growth is harder than final size")
    for ax in axes:ax.grid(axis="y",alpha=.15)
    fig.suptitle("Does baseline composition and nutrition improve prediction?",fontweight="bold",fontsize=15,y=.99)
    fig.text(.01,.015,"67 participants; 20 repeats of 5-fold CV. Ridge tuned inside training folds. Error bars = split SD, not population confidence intervals.",fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(ROOT/"figures/04_model_comparison.png",dpi=200);plt.close(fig)

    coef=coefficients[(coefficients.model=="extended_ols")&(coefficients.term.isin(ADDED))].set_index("term").loc[ADDED]
    fig,ax=plt.subplots(figsize=(9,4.8));y=np.arange(len(ADDED))
    beta=coef.beta_cm.to_numpy();low=coef.ci95_low_cm.to_numpy();high=coef.ci95_high_cm.to_numpy()
    ax.errorbar(beta,y,xerr=np.vstack([beta-low,high-beta]),fmt="o",color=teal,ecolor=navy,capsize=4,markersize=6)
    ax.axvline(0,color="#697889",linestyle="--")
    ax.set(yticks=y,yticklabels=["Lean mass (+10 kg)","Fat mass (+10 kg)","Protein (+25 g/day)","Energy (+500 kcal/day)"],
           xlabel="Adjusted association with POST thickness (cm)",title="Added baseline features: coefficients and HC3 95% intervals")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
    fig.text(.01,.02,"Full-sample OLS adjusted for initial thickness and all four added features. t-reference intervals are approximate; associations are not causal effects.",fontsize=8)
    fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(ROOT/"figures/05_association_coefficients.png",dpi=200);plt.close(fig)

    fit=fits["extended_ols"];fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    axes[0].scatter(fit["fitted"],fit["residual"],color=teal,s=25,alpha=.8)
    axes[0].axhline(0,color=navy,linewidth=.8)
    axes[0].set(xlabel="Fitted POST thickness (cm)",ylabel="Residual (cm)",title="Residuals versus fitted")
    qq=stats.probplot(fit["residual"],dist="norm")
    theoretical,ordered=qq[0];slope,intercept,_=qq[1]
    axes[1].scatter(theoretical,ordered,color=teal,s=20);axes[1].plot(theoretical,intercept+slope*theoretical,color=navy)
    axes[1].set(xlabel="Theoretical normal quantile",ylabel="Ordered residual (cm)",title="Residual Q-Q plot")
    axes[2].scatter(fit["leverage"],fit["cook"],color=teal,s=25)
    axes[2].axhline(4/len(diagnostics),color=navy,linestyle="--",label="Cook's screening 4/n")
    for _,r in diagnostics.nlargest(2,"cooks_distance").iterrows():
        axes[2].annotate(r.participant_id,(r.leverage,r.cooks_distance),xytext=(3,3),textcoords="offset points",fontsize=8)
    axes[2].set(xlabel="Leverage",ylabel="Cook's distance",title="Influence screening (no deletions)")
    axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle("Primary extended linear model: diagnostics",fontweight="bold",fontsize=14)
    fig.text(.01,.015,"All 67 participants retained. Screening thresholds identify observations for review and do not prove a data error.",fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.94]);fig.savefig(ROOT/"figures/06_model_diagnostics.png",dpi=200);plt.close(fig)


def write_reports(d,fits,coef,association_summary,diagnostics,vif,summary,comparisons,ridge):
    delta_mae=float(summary.loc["extended_ols","mae_cm_mean"]-summary.loc["baseline_ols","mae_cm_mean"])
    delta_mae_ridge=float(summary.loc["extended_ridge","mae_cm_mean"]-summary.loc["baseline_ols","mae_cm_mean"])
    pair=comparisons[(comparisons.extended=="extended_ols")&(comparisons.baseline=="baseline_ols")]
    ridge_pair=comparisons[(comparisons.extended=="extended_ridge")&(comparisons.baseline=="baseline_ols")]
    primary=association_summary.set_index("model").loc["extended_ols"]
    baseline_assoc=association_summary.set_index("model").loc["baseline_ols"]
    flagged=diagnostics[diagnostics.flag_cook_4_over_n|diagnostics.flag_leverage_2p_over_n]
    evidence={
        "analysis_specification_date":"2026-10-13","n":len(d),"cv_repeats":REPEATS,"outer_folds":FOLDS,"seed":SEED,
        "baseline_mean_mae_cm":float(summary.loc["baseline_ols","mae_cm_mean"]),
        "extended_mean_mae_cm":float(summary.loc["extended_ols","mae_cm_mean"]),
        "ridge_mean_mae_cm":float(summary.loc["extended_ridge","mae_cm_mean"]),
        "extended_minus_baseline_mean_mae_cm":delta_mae,
        "ridge_minus_baseline_mean_mae_cm":delta_mae_ridge,
        "extended_improved_mae_repeats":int((pair.mae_cm_difference_extended_minus_baseline<0).sum()),
        "ridge_improved_mae_repeats":int((ridge_pair.mae_cm_difference_extended_minus_baseline<0).sum()),
        "primary_added_block_hc3_wald_p_approx":float(primary.added_block_wald_p_approx),
        "max_primary_vif":float(vif.vif.max()),
        "cook_flagged_ids":diagnostics.loc[diagnostics.flag_cook_4_over_n,"participant_id"].tolist(),
        "leverage_flagged_ids":diagnostics.loc[diagnostics.flag_leverage_2p_over_n,"participant_id"].tolist(),
        "ridge_alpha_selection_counts":{str(k):int(v) for k,v in ridge.chosen_alpha.value_counts().sort_index().items()},
        "no_features_selected_after_results":True,"no_participants_removed":True,
        "hc3_verified_by_independent_delete_one_fits":True,
        "runtime":{"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__},
        "core_csv_sha256":hashlib.sha256((ROOT/"data/processed/participants_core.csv").read_bytes()).hexdigest(),
    }
    (RESULTS/"stage2_summary.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    lines=["# 第二阶段：基线身体组成与营养是否改善增肌预测？","",
           "本报告来自实际运行的模型。67 名参与者全部保留；主要变量与第一阶段方案一致，没有根据结果挑选变量。","",
           "## 1. 本次完成的分析","",
           "- 关联分析：POST 肌肉厚度为结果，调整 PRE 厚度，并纳入基线瘦体重、脂肪质量、蛋白质和热量。",
           "- 系数使用 HC3 稳健标准误；95% 区间和 p 值采用残差自由度的近似 t 参考。用逐人删除的独立最小二乘拟合验证 HC3 实现。",
           "- 预测分析：同样的 20 次重复 × 5 折交叉验证；每次让全部 67 人各得到一次来自未见过该人的模型的预测。",
           "- Ridge 在外层训练数据内部再做 5 折调参，标准化与调参均不接触外层测试数据。",
           "- 敏感性分析：年龄调整，以及年龄加补充剂代码分类调整。原始数字代码未替换成猜测的补充剂名称。","",
           "## 2. 对未参与拟合者的预测结果","",
           "表中为每次重复汇总所有留出预测后的指标，再对 20 次取均值。误差单位为 cm。重复次数不增加独立样本量。","",
           "| 模型 | MAE 均值 ± 划分标准差 | RMSE 均值 | POST R² 均值 | 观察到的变化 R² 均值 |",
           "|---|---:|---:|---:|---:|"]
    cn={"reference_mean_post":"训练集 POST 均值参考","reference_mean_change":"PRE 加训练集平均变化参考",
        "baseline_ols":"只用初始厚度","extended_ols":"初始厚度 + 身体组成 + 营养","extended_ridge":"相同五个特征 + Ridge 正则化"}
    for m in MAIN_MODELS:
        s=summary.loc[m]
        lines.append(f'| {cn[m]} | {s.mae_cm_mean:.4f} ± {s.mae_cm_sd:.4f} | {s.rmse_cm_mean:.4f} | {s.r2_post_mean:.3f} | {s.r2_observed_change_mean:.3f} |')
    if delta_mae >= 0 and delta_mae_ridge >= 0:
        lines += ["", "**本次结论：按照固定特征和验证方案，加入身体组成、基线蛋白质和热量没有改善留出预测，普通线性模型和 Ridge 的平均误差均略大。** 这不等于证明营养没有作用，也不证明这些变量在更大的数据或其他测量设计下没有价值。"]
    phrase="降低" if delta_mae<0 else "增加"
    phrase_r="降低" if delta_mae_ridge<0 else "增加"
    lines += ["",f'加入四个特征的普通线性模型，相比只用初始厚度，平均 MAE {phrase} {abs(delta_mae):.4f} cm（{abs(delta_mae)*10:.3f} mm）。20 次划分中有 {evidence["extended_improved_mae_repeats"]} 次 MAE 更低。',
              f'相同特征的 Ridge 模型，平均 MAE {phrase_r} {abs(delta_mae_ridge):.4f} cm；20 次中有 {evidence["ridge_improved_mae_repeats"]} 次 MAE 更低。',
              "这些次数和划分标准差不是独立试验、不是显著性检验，也不是总体性能置信区间。", "",
              "![模型比较](../../figures/04_model_comparison.png)","",
              "POST 厚度预测得好，可能主要来自初始肌肉大小。变化 R²则考察 POST-PRE 的预测；它与 POST 的 R²不同。两者的 MAE/RMSE相同，因为实测和预测都减去了同一个人的 PRE 值。负的变化 R²说明该验证误差比用全部样本的真实变化均值作数学参考更大；实际可部署的训练集平均变化参考已另外交叉验证。","",
              "## 3. 全样本的条件关联","",
              "| 基线特征变化单位 | POST 厚度系数（cm） | 近似 HC3 95% 区间 | 未校正 p 值 |",
              "|---|---:|---|---:|"]
    term_cn={"pre_vl_cm":"初始厚度 +0.1 cm","pre_lean_mass_kg":"DXA 瘦体重 +10 kg","pre_fat_mass_kg":"脂肪质量 +10 kg",
             "pre_protein_g_day":"蛋白质 +25 g/day","pre_energy_kcal_day":"热量 +500 kcal/day"}
    c=coef[coef.model=="extended_ols"].set_index("term")
    for k in FULL:
        s=c.loc[k]
        p_text="<0.0001" if s.p_value_unadjusted<.0001 else f"{s.p_value_unadjusted:.4f}"
        lines.append(f'| {term_cn[k]} | {s.beta_cm:.4f} | [{s.ci95_low_cm:.4f}, {s.ci95_high_cm:.4f}] | {p_text} |')
    lines += ["",f'四个新增特征的联合 HC3 Wald F 近似检验 p = {primary.added_block_wald_p_approx:.4f}。该检验用于这批数据的条件关联，并不保证预测改善，更不证明改变饮食会造成对应增肌。',
              f'样本内 R² 从基线模型的 {baseline_assoc.r2:.3f} 增至扩展模型的 {primary.r2:.3f}，但验证误差并没有对应改善。新增变量让样本内拟合更好，本身不能作为预测有效的证据。',
              "个别特征的 p 值未作多重比较校正。不要用单个 p 值决定加删变量或宣称发现机制。","",
              "![条件关联系数](../../figures/05_association_coefficients.png)","",
              "## 4. 模型诊断与稳健性","",
              "| 特征 | VIF |","|---|---:|"]
    for _,r in vif.iterrows():lines.append(f'| {term_cn[r.variable]} | {r.vif:.3f} |')
    lines += ["",f'Cook 距离筛查超过 4/n 的参与者：{", ".join(evidence["cook_flagged_ids"]) or "无"}。',
              f'杠杆值超过 2p/n 的参与者：{", ".join(evidence["leverage_flagged_ids"]) or "无"}。',
              "筛查标记并不表示数据错误，主要分析没有删除任何人。每次删除一人的系数结果已完整导出，可检查方向和大小是否稳定。","",
              "![模型诊断](../../figures/06_model_diagnostics.png)","",
              "## 5. 预先声明的敏感性比较","",
              "下表对基线和扩展模型使用同样的年龄/组别控制。数值为扩展模型减基线模型，MAE/RMSE 负值表示改善。","",
              "| 比较 | MAE 差均值（cm） | RMSE 差均值（cm） | POST R² 差均值 |","|---|---:|---:|---:|"]
    for a,b,label in [("extended_age_ols","baseline_age_ols","两者均调整年龄"),
                      ("extended_age_group_ols","baseline_age_group_ols","两者均调整年龄及补充剂类别")]:
        g=comparisons[(comparisons.extended==a)&(comparisons.baseline==b)]
        lines.append(f'| {label} | {g.mae_cm_difference_extended_minus_baseline.mean():.4f} | {g.rmse_cm_difference_extended_minus_baseline.mean():.4f} | {g.r2_post_difference_extended_minus_baseline.mean():.3f} |')
    lines += ["","## 6. 结论边界","",
              "当前能回答的是：在这批 67 人、这套固定变量和验证方案下，新增信息带来多大的条件关联与预测变化。不能把结果扩展为普遍的蛋白质无效/有效结论。",
              "样本小、自报饮食有误差、没有实际体重列、没有非训练对照、没有外部验证，且群体限于年轻未训练男性。POST 饮食、累计训练量和分子标志物没有放进训练前预测。",
              "本次模型不能用于给个人开增肌处方。统计区间与交叉验证也不能把测量波动分离成稳定的先天增肌体质。","",
              "## 7. 如何在你的电脑重现","",
              "在解压后的新项目主文件夹打开 PowerShell。如果尚无 .venv 文件夹，先执行 `python -m venv .venv` 创建环境，再依次运行：","","```powershell",
              ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
              ".\\.venv\\Scripts\\python.exe src\\run_models.py","```","",
              "该脚本先从原始 Excel 重新生成第一阶段结果，然后完成关联、验证、诊断和图表。运行完成会显示 `Complete: association models, nested cross-validation, diagnostics and reports.`。",
              "所有模型指标、划分、留出预测、Ridge 搜索结果和系数保存在 results/models/。","",
              "## 来源与署名","",
              "原始数据和实验：Mobley et al. (2018), https://doi.org/10.1371/journal.pone.0195203 。原始数据作者不是本项目新脚本的作者。",
              "本项目清洗、关联分析与验证代码为本项目新编写，使用 AI 辅助生成。使用者应理解、审查并如实说明自己的贡献。",
              "HC3 公式：https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html",
              "scikit-learn 验证方法：https://scikit-learn.org/1.8/modules/cross_validation.html","",
              "说明：全样本拟合参数用于关联解释；交叉验证误差来自独立留出拟合，不是全样本训练误差。"]
    (RESULTS/"Stage2_Model_Report_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Standalone English results for eventual repository review.
    en=["# Stage 2 model results","","Secondary analysis of 67 participants. All five primary predictors were declared before fitting new models.","",
        "| Model | Mean OOF MAE (cm) | Mean OOF RMSE (cm) | Mean POST R² | Mean change R² |","|---|---:|---:|---:|---:|"]
    for m in MAIN_MODELS:
        s=summary.loc[m];en.append(f'| {LABELS[m]} | {s.mae_cm_mean:.4f} | {s.rmse_cm_mean:.4f} | {s.r2_post_mean:.3f} | {s.r2_observed_change_mean:.3f} |')
    en += ["",f'Extended OLS minus baseline OLS mean MAE: {delta_mae:+.4f} cm. Tuned ridge minus baseline: {delta_mae_ridge:+.4f} cm.',
           "",("Under this fixed specification, neither the extended linear model nor tuned ridge improved mean held-out MAE over the initial-thickness-only model. This is not evidence that nutrition has no physiological effect." if delta_mae>=0 and delta_mae_ridge>=0 else "Interpret improvement relative to the matched baseline and report its size and uncertainty; it does not establish causality."),
           "", "Metrics pool the five out-of-fold partitions per repeat, then summarize 20 repeats. This is not the same as averaging fold R². The repeats reuse the same people; split SD is not a population confidence interval.",
           "", "Ridge scaling and tuning used only outer training data with inner 5-fold validation. The same splits were used for every model. Both age and age-plus-categorical-group sensitivity comparisons are exported.",
           "", "HC3 covariance was implemented from the documented formula and verified by independent delete-one least-squares fits. t-reference intervals and block Wald F tests are approximate; individual p-values are not multiplicity adjusted.",
           "", "The reported coefficients are conditional associations, not intervention effects. High predictability of POST size does not establish high predictability of growth. No external validation, personal prescription, or stable responder-class claim is made.",
           "", "Raw data: Mobley et al. (2018), PLOS ONE 13(4):e0195203, S2. https://doi.org/10.1371/journal.pone.0195203. New analysis code is project-specific and AI-assisted."]
    (RESULTS/"Stage2_Model_Results_EN.md").write_text("\n".join(en)+"\n",encoding="utf-8")
    return evidence


def main():
    print("Reproducing Stage 1 from the verified raw source...",flush=True)
    run_eda.main()
    RESULTS.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/"data/processed/participants_core.csv",dtype={"supplement_code":"string"})
    assert len(d)==67 and d.participant_id.nunique()==67
    assert d[FULL+["post_vl_cm"]].notna().all().all()
    fits,coef,assoc,diagnostics,vif=fit_associations(d)
    print("Association models and HC3 independent checks complete.",flush=True)
    oof,score_df,summary,comparisons,ridge=run_cv(d)
    make_figures(fits,coef,diagnostics,summary,score_df)
    evidence=write_reports(d,fits,coef,assoc,diagnostics,vif,summary,comparisons,ridge)
    import verify_models
    verify_models.main()
    print(json.dumps(evidence,indent=2))
    print("Complete: association models, nested cross-validation, diagnostics and reports.",flush=True)


if __name__=="__main__":
    main()
