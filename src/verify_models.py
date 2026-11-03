"""Independent checks of exported splits, metrics, OLS, and nested ridge.

Run after run_models.py: python src/verify_models.py
This checks computed evidence, not just whether output files exist.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/models"
FEATURES=["pre_vl_cm","pre_lean_mass_kg","pre_fat_mass_kg","pre_protein_g_day","pre_energy_kcal_day"]


def direct_ridge(train_x,train_y,test_x,alpha):
    mean=train_x.mean(axis=0);scale=train_x.std(axis=0,ddof=0)
    z=(train_x-mean)/scale
    coefficient=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(train_y-train_y.mean()))
    return train_y.mean()+((test_x-mean)/scale)@coefficient


def main():
    d=pd.read_csv(ROOT/"data/processed/participants_core.csv").set_index("participant_id")
    outer=pd.read_csv(OUT/"cv_outer_splits.csv")
    inner=pd.read_csv(OUT/"cv_inner_splits.csv")
    oof=pd.read_csv(OUT/"cv_out_of_fold_predictions.csv")
    scores=pd.read_csv(OUT/"cv_repeat_metrics.csv")
    choices=pd.read_csv(OUT/"cv_ridge_selection.csv")
    grid=pd.read_csv(OUT/"cv_ridge_grid_scores.csv")
    assert len(d)==67 and d.index.is_unique
    assert (oof.groupby(["repeat","model","participant_id"]).size()==1).all()
    assert (outer[outer.role=='test'].groupby(["repeat","participant_id"]).size()==1).all()
    for (repeat,fold),g in outer.groupby(["repeat","fold"]):
        tr=set(g[g.role=='train'].participant_id);te=set(g[g.role=='test'].participant_id)
        assert not tr&te and tr|te==set(d.index)
        test_predictions=oof[(oof.repeat==repeat)&(oof.fold==fold)]
        assert set(test_predictions.participant_id)==te
        for _,ig in inner[(inner.repeat==repeat)&(inner.outer_fold==fold)].groupby('inner_fold'):
            it=set(ig[ig.role=='train'].participant_id);iv=set(ig[ig.role=='validation'].participant_id)
            assert not it&iv and it|iv==tr and not (it|iv)&te
    for (repeat,model),g in oof.groupby(["repeat","model"]):
        err=g.observed_post_cm.to_numpy()-g.predicted_post_cm.to_numpy()
        mae=np.abs(err).mean();rmse=np.sqrt(np.mean(err**2))
        growth=g.observed_change_cm.to_numpy()
        r2=1-np.sum(err**2)/np.sum((growth-growth.mean())**2)
        s=scores[(scores.repeat==repeat)&(scores.model==model)].iloc[0]
        assert np.isclose(s.mae_cm,mae,atol=1e-12) and np.isclose(s.rmse_cm,rmse,atol=1e-12)
        assert np.isclose(s.r2_observed_change,r2,atol=1e-12)
    # SciPy's independent simple-regression route matches exported baseline coefficients.
    coef=pd.read_csv(OUT/"association_coefficients_hc3.csv")
    baseline=stats.linregress(d.pre_vl_cm,d.post_vl_cm)
    b=coef[(coef.model=='baseline_ols')&(coef.term=='pre_vl_cm')].iloc[0]
    assert np.isclose(b.beta_cm,baseline.slope*.1,atol=1e-12)
    # Independently reproduce first outer split and its four alpha candidates.
    g=outer[(outer.repeat==1)&(outer.fold==1)]
    tr=g[g.role=='train'].participant_id.tolist();te=g[g.role=='test'].participant_id.tolist()
    X=d[FEATURES];y=d.post_vl_cm
    first=inner[(inner.repeat==1)&(inner.outer_fold==1)]
    independent_scores={}
    for alpha in [.1,1.,10.,100.]:
        errors=[]
        for _,ig in first.groupby('inner_fold'):
            it=ig[ig.role=='train'].participant_id.tolist();iv=ig[ig.role=='validation'].participant_id.tolist()
            pred=direct_ridge(X.loc[it].to_numpy(),y.loc[it].to_numpy(),X.loc[iv].to_numpy(),alpha)
            errors.append(np.mean((y.loc[iv].to_numpy()-pred)**2))
        independent_scores[alpha]=float(np.mean(errors))
        row=grid[(grid.repeat==1)&(grid.fold==1)&np.isclose(grid.alpha,alpha)].iloc[0]
        assert np.isclose(row.mean_inner_mse_cm2,independent_scores[alpha],atol=1e-12)
    chosen=min(independent_scores,key=independent_scores.get)
    exported_alpha=float(choices[(choices.repeat==1)&(choices.fold==1)].chosen_alpha.iloc[0])
    assert chosen==exported_alpha
    pred=direct_ridge(X.loc[tr].to_numpy(),y.loc[tr].to_numpy(),X.loc[te].to_numpy(),chosen)
    saved=oof[(oof.repeat==1)&(oof.fold==1)&(oof.model=='extended_ridge')].set_index('participant_id').loc[te]
    assert np.allclose(pred,saved.predicted_post_cm.to_numpy(),atol=1e-10)
    results={"participant_n":len(d),"outer_train_test_disjoint_all_100_splits":True,
             "inner_splits_contained_in_outer_training_all_500_splits":True,
             "one_oof_prediction_per_participant_per_repeat_per_model":True,
             "exported_repeat_metrics_recomputed_independently":True,
             "baseline_coefficient_matches_scipy_linregress":True,
             "first_outer_ridge_grid_scores_and_predictions_match_closed_form":True,
             "first_outer_independent_ridge_mse":independent_scores,
             "first_outer_selected_alpha":chosen}
    (OUT/"verification_checks.json").write_text(json.dumps(results,indent=2)+"\n",encoding='utf-8')
    print(json.dumps(results,indent=2))
    print("Complete: independent model and validation checks passed.")


if __name__=='__main__':main()
