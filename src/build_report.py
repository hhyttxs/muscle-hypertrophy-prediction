"""Build the six-page research report from exported Stage 2 evidence.

Optional dependency: reportlab. Core modeling does not require it.
Run after run_models.py: python src/build_report.py
"""
from pathlib import Path
from html import escape
import json
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'results/models'
NAVY=colors.HexColor('#17324D');TEAL=colors.HexColor('#168B8A')


def fonts():
    # Use an available full-Unicode font on Linux; keep a built-in fallback for Windows.
    roots=[Path('/usr/share/fonts/truetype/dejavu'),Path('C:/Windows/Fonts')]
    for root in roots:
        normal=root/'DejaVuSans.ttf';bold=root/'DejaVuSans-Bold.ttf'
        if normal.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont('Study',str(normal)))
            pdfmetrics.registerFont(TTFont('StudyBold',str(bold)))
            pdfmetrics.registerFontFamily('Study',normal='Study',bold='StudyBold',italic='Study',boldItalic='StudyBold')
            return 'Study','StudyBold'
    return 'Helvetica','Helvetica-Bold'


def main():
    font,bold=fonts()
    s={
        'body':ParagraphStyle('body',fontName=font,fontSize=9.3,leading=14,spaceAfter=8,textColor=NAVY),
        'small':ParagraphStyle('small',fontName=font,fontSize=7.8,leading=11,spaceAfter=7,textColor=colors.HexColor('#415366')),
        'h1':ParagraphStyle('h1',fontName=bold,fontSize=20,leading=25,spaceAfter=14,textColor=NAVY),
        'h2':ParagraphStyle('h2',fontName=bold,fontSize=12,leading=17,spaceBefore=8,spaceAfter=8,textColor=TEAL),
        'table':ParagraphStyle('table',fontName=font,fontSize=8,leading=11,textColor=NAVY),
    }
    summary=pd.read_csv(M/'cv_metric_summary.csv').set_index('model')
    assoc=pd.read_csv(M/'association_model_summary.csv').set_index('model')
    coef=pd.read_csv(M/'association_coefficients_hc3.csv')
    coef=coef[coef.model=='extended_ols'].set_index('term')
    eda=json.loads((ROOT/'results/eda_summary.json').read_text())
    proof=json.loads((M/'verification_checks.json').read_text())
    source=json.loads((ROOT/'data/source_manifest.json').read_text())
    paired=pd.read_csv(M/'cv_paired_model_differences.csv')
    story=[]
    def p(text,style='body'):story.append(Paragraph(text,s[style]))
    def h(text):p(text,'h2')
    def table(rows,widths):
        cells=[[Paragraph(escape(str(v)),s['table']) for v in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9F1F5')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,0),(-1,0),.7,NAVY),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D8E0E6'))]))
        story.extend([t,Spacer(1,10)])
    def fig(name,height,caption):
        im=Image(str(ROOT/'figures'/name));ratio=im.imageWidth/im.imageHeight
        width=min(479,height*ratio);im.drawWidth=width;im.drawHeight=width/ratio
        story.append(im);p(caption,'small')
    def page():story.append(PageBreak())

    p('Baseline Nutrition and Body Composition in Muscle Thickness Prediction','h1')
    p('A reproducible secondary analysis of a 12-week resistance-training dataset','h2')
    p('Final review edition | Analysis and internal validation complete | Prepared 21 October 2026','small')
    h('Abstract')
    p('This study asked whether pre-training body composition, protein intake and energy intake add explanatory and predictive information beyond initial vastus lateralis thickness. Public participant-level data from Mobley et al. (2018) were analyzed using a fixed five-predictor specification. All 67 participants had complete primary fields. Baseline-adjusted linear association models used HC3 covariance. Prediction was evaluated with 20 repeats of 5-fold participant-level cross-validation, with inner validation for ridge regularization. Mean absolute error was 0.1861 cm for initial thickness alone, 0.1949 cm after adding composition and nutrition, and 0.1966 cm for tuned ridge. The approximate joint HC3 Wald test for the four added features gave p=0.589. Under this specification, added baseline measurements did not improve held-out prediction. The finding concerns incremental information in this sample; it does not establish that nutrition has no physiological effect.')
    h('Research question')
    p('Do baseline DXA lean mass, fat mass, protein intake and energy intake improve prediction of 12-week muscle thickness beyond its initial value? Association and prediction are assessed separately. The data preparation, model comparison, internal validation and report are complete; GitHub publication and external validation are not completed.')
    table([['Study item','Definition'],['Participants','67 young, previously untrained men'],['Design','Paired PRE and POST measurements over 12 weeks'],['Primary measurement','Ultrasound vastus lateralis thickness, in cm'],['Primary sample','67 people; one row per person; no primary missing values'],['Interpretation','Secondary observational analysis within an existing training/supplement setting']], [119,360])
    p('Original experiment and data: Mobley et al. (2018), PLOS ONE 13(4):e0195203, S2 File [1]. The original investigators collected the data; this project supplies new analysis code and documentation.','small')
    page()

    p('Methods and fixed analysis design','h1')
    h('Data preparation')
    p('The original XLSX was retained unchanged and checked by SHA-256. Participant IDs matching MPS followed by digits identified real records; author summary rows, statistical tests and legends were excluded. Source row numbers were preserved. T1 denotes PRE and T39 denotes POST, rather than week 39. No imputation, extreme-value deletion or outcome-driven feature selection was performed. The analysis protocol was written after reading the original paper, before fitting the new models; it was not a registered preregistration.')
    h('Outcome and predictor timing')
    p('The outcome was POST thickness, adjusted for PRE thickness. The four added predictors were PRE lean mass (kg), fat mass (kg), protein (g/day) and energy (kcal/day). Diet fields represent four-day self-reports [1,2]. Age and supplement-code controls were sensitivity analyses. POST nutrition, total intervention volume, POST biomarkers and outcome-derived response classes were excluded from pre-training prediction. Direct body weight was absent, so protein per kg body weight was not calculated.')
    table([['Model','Inputs or prediction rule'],['Mean POST reference','Training-fold mean POST thickness'],['Mean-change reference','Observed PRE plus training-fold average change'],['Baseline linear model','PRE thickness only'],['Extended linear model','PRE thickness plus the four added features'],['Extended ridge','Same five predictors; alpha selected internally']], [153,326])
    h('Validation and inference')
    p('The same outer folds were used for every model (20 repeats x 5 folds, seed 20261011). Each repeat produced one held-out prediction per participant and model. Metrics pooled the five held-out partitions within a repeat, then summarized 20 repeats; fold metrics were also exported. Ridge used inner 5-fold validation with alpha in {0.1, 1, 10, 100}. Scaling and encoding were fitted to training data only [3].')
    p('OLS coefficients used HC3 covariance [4], independently verified by delete-one least-squares fits. Coefficient intervals used a t reference with n-p residual degrees of freedom; the four-feature block used an approximate Wald F reference. These references are approximate. Individual p-values were not multiplicity adjusted. Split-to-split SDs are not population confidence intervals.')
    page()

    p('Prediction results','h1')
    names=[('reference_mean_post','Mean POST reference'),('reference_mean_change','Mean-change reference'),('baseline_ols','Initial thickness only'),('extended_ols','Add composition + nutrition'),('extended_ridge','Same features, tuned ridge')]
    rows=[['Model','MAE (cm)','RMSE (cm)','POST R2','Change R2']]
    for m,name in names:
        v=summary.loc[m];rows.append([name,f'{v.mae_cm_mean:.4f}',f'{v.rmse_cm_mean:.4f}',f'{v.r2_post_mean:.3f}',f'{v.r2_observed_change_mean:.3f}'])
    table(rows,[205,66,76,66,66])
    p('Metrics are means across 20 repeats of pooled held-out predictions. Lower MAE and RMSE are better. All scores use the same 67 participants. The repeated splits do not increase the independent sample size.','small')
    fig('04_model_comparison.png',205,'Figure 1. Held-out prediction errors and observed-change R-squared. Error bars show split SD, not a population confidence interval.')
    h('Incremental prediction')
    increase=summary.loc['extended_ols','mae_cm_mean']-summary.loc['baseline_ols','mae_cm_mean']
    p(f'The extended linear model increased MAE by {increase:.4f} cm ({increase*10:.3f} mm). Its MAE was lower than baseline in only 1 of 20 repeats; ridge was lower in 0 of 20. These counts are split sensitivity, not independent trials or a formal test of generalization superiority. No external validation was available.')
    h('Final size is not the same target as growth')
    p('The baseline model had mean POST R-squared 0.457, but observed-change R-squared only 0.049. Predicted growth equals predicted POST minus observed PRE. This subtraction leaves MAE and RMSE unchanged, while R-squared uses a different variance denominator. Thus, substantial predictability of final size can coexist with weak prediction of individual growth.')
    page()

    p('Conditional associations','h1')
    p('The extended OLS model adjusted for all five baseline features simultaneously. Coefficients below describe conditional associations with POST thickness; they are not effects of changing an individual\'s diet.')
    labels=[('pre_vl_cm','PRE thickness +0.1 cm'),('pre_lean_mass_kg','Lean mass +10 kg'),('pre_fat_mass_kg','Fat mass +10 kg'),('pre_protein_g_day','Protein +25 g/day'),('pre_energy_kcal_day','Energy +500 kcal/day')]
    rows=[['Baseline contrast','Coefficient (cm)','HC3 95% interval','Unadjusted p']]
    for key,label in labels:
        v=coef.loc[key];pv='<0.0001' if v.p_value_unadjusted<.0001 else f'{v.p_value_unadjusted:.4f}'
        rows.append([label,f'{v.beta_cm:.4f}',f'[{v.ci95_low_cm:.4f}, {v.ci95_high_cm:.4f}]',pv])
    table(rows,[161,95,143,80])
    p('The four added features jointly had approximate HC3 Wald p=0.589. Each added-feature interval included zero. These wide intervals are consistent with substantial uncertainty and do not establish equivalence or zero biological effect.')
    fig('05_association_coefficients.png',235,'Figure 2. Added-feature estimates and approximate HC3 intervals from the full-sample linear model, conditional on PRE thickness and the other included variables.')
    h('Fit versus validation')
    p(f'Full-sample R-squared increased from {assoc.loc["baseline_ols","r2"]:.3f} to {assoc.loc["extended_ols","r2"]:.3f}, while adjusted R-squared fell from {assoc.loc["baseline_ols","adj_r2"]:.3f} to {assoc.loc["extended_ols","adj_r2"]:.3f}. The observed improvement in apparent fit did not translate into improved held-out prediction. These are different evaluation questions.')
    page()

    p('Robustness and interpretation','h1')
    h('Diagnostics and declared sensitivities')
    p('All participants were retained. Five exceeded the Cook distance screening threshold 4/n and five exceeded leverage 2p/n; these were review flags, not proof of errors. VIFs ranged from 1.014 to 2.434. Leave-one-participant-out estimates were exported. The four added-feature point-estimate directions remained the same across these omissions, although their full-sample uncertainty intervals included zero.')
    rows=[['Matched adjustment','MAE difference (cm)','POST R2 difference']]
    for a,b,label in [('extended_age_ols','baseline_age_ols','Both models adjust for age'),('extended_age_group_ols','baseline_age_group_ols','Both adjust for age + group')]:
        g=paired[(paired.extended==a)&(paired.baseline==b)]
        rows.append([label,f'{g.mae_cm_difference_extended_minus_baseline.mean():+.4f}',f'{g.r2_post_difference_extended_minus_baseline.mean():+.3f}'])
    table(rows,[239,126,114])
    p('Differences are extended minus baseline; positive MAE indicates more error. The supplement labels behind the five numeric group codes were not verified, so codes were categorical controls without named-supplement interpretations.','small')
    fig('06_model_diagnostics.png',160,'Figure 3. Residual, distribution and influence checks for the extended OLS model. Review flags did not alter the primary sample.')
    h('Limitations')
    for text in [
        'A small, selected sample limits precision and generalization to women, older adults or trained populations.',
        'Brief self-reported baseline diet is not a measurement of habitual intake throughout training. Supplement accounting for POST nutrition remains unresolved.',
        'DXA lean mass is not pure skeletal muscle mass. No actual height/body-weight field was available for normalized indices.',
        'No non-training control or independent repeated training cycle is present in this analysis. Observed variation mixes physiological change, day-to-day variation and measurement error.',
        'Repeated internal validation is not external validation. This study does not establish an individual training-response phenotype or an optimal protein prescription.'
    ]:p(escape(text))
    page()

    p('Reproducibility and conclusion','h1')
    h('What was verified')
    p('All 100 outer train/test splits were disjoint. All 500 inner splits remained within their outer training data. Saved metrics were recomputed independently. The baseline coefficient matched SciPy simple regression. The first nested ridge grid and held-out predictions matched a direct closed-form calculation. A clean-extraction rerun reproduced the key Stage 2 outputs.')
    p('The user independently ran the package on Windows with Python 3.12.10 and the pinned scientific libraries. The supplied log reproduced the primary MAEs, group counts and verification checks. Its different cleaned-CSV SHA-256 was reproduced exactly by converting local LF line endings to Windows CRLF. Version 3 explicitly writes LF CSV line endings; this is a formatting fix, not a model or data-value change.')
    h('Reproduce the analysis')
    p('In the extracted project folder, create a Python environment and install requirements.txt. Run src/run_models.py. It regenerates Stage 1, all declared models, diagnostics and verification. Review results/models/Stage2_Model_Report_CN.md and the machine-readable metrics. src/build_report.py separately builds this PDF and requires requirements-report.txt.')
    h('Conclusion')
    p('Within this sample and fixed specification, baseline body composition, protein and energy did not add useful held-out predictive information beyond initial muscle thickness. The appropriate result is a documented limit of the available measurements and model, not a universal nutritional null effect. Further work would require justified improvements to measurement or genuinely independent data, with new exploratory decisions recorded separately.')
    h('Data and code provenance')
    p('Original data and experimental design belong to the authors of [1]. The source XLSX is unchanged; processing and figures are derived analyses. The project-specific Python scripts were newly developed with AI assistance; original-investigator analysis code was not copied. A public-data secondary analysis is not a newly collected clinical study. Contributor roles should reflect work actually understood and performed.')
    h('References')
    refs=[
        '[1] Mobley CB et al. (2018). Biomarkers associated with low, moderate, and high vastus lateralis muscle hypertrophy following 12 weeks of resistance training. PLOS ONE 13(4):e0195203. doi:10.1371/journal.pone.0195203. S2 File: Raw data. CC BY 4.0.',
        '[2] Mobley CB et al. (2017). Effects of whey, soy or leucine supplementation with 12 weeks of resistance training on strength, body composition, and skeletal muscle and adipose tissue histological attributes in college-aged males. Nutrients 9(9):972. doi:10.3390/nu9090972.',
        '[3] scikit-learn 1.8 documentation. Cross-validation and training-only transformations. https://scikit-learn.org/1.8/modules/cross_validation.html',
        '[4] statsmodels documentation. OLSResults.HC3_se: documented HC3 covariance definition. https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html'
    ]
    for ref in refs:p(escape(ref),'small')

    output=ROOT/'reports/Hypertrophy_Research_Report_FINAL_EN.pdf'
    output.parent.mkdir(exist_ok=True)
    def decorate(c,doc):
        c.setStrokeColor(colors.HexColor('#D8E0E6'));c.line(58,44,A4[0]-58,44)
        c.setFont(font,7.3);c.setFillColor(NAVY)
        c.drawString(58,31,'Muscle thickness prediction | Secondary analysis | 21 October 2026')
        c.drawRightString(A4[0]-58,31,str(doc.page))
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=58,leftMargin=58,topMargin=49,bottomMargin=57,
                          title='Baseline Nutrition and Body Composition in Muscle Thickness Prediction',
                          author='Independent portfolio study; AI-assisted analysis')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(output)


if __name__=='__main__':main()
