"""Create the Chinese final review report using exported numerical evidence."""
from pathlib import Path
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

ROOT=Path(__file__).resolve().parents[1]


def main():
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    navy=colors.HexColor('#17324D');teal=colors.HexColor('#168B8A')
    styles={
        'title':ParagraphStyle('title',fontName='STSong-Light',fontSize=21,leading=29,textColor=navy,spaceAfter=14,wordWrap='CJK'),
        'h':ParagraphStyle('h',fontName='STSong-Light',fontSize=13,leading=19,textColor=teal,spaceBefore=10,spaceAfter=8,wordWrap='CJK'),
        'body':ParagraphStyle('body',fontName='STSong-Light',fontSize=10.5,leading=18,textColor=navy,spaceAfter=9,wordWrap='CJK'),
        'small':ParagraphStyle('small',fontName='STSong-Light',fontSize=8.5,leading=13,textColor=navy,spaceAfter=7,wordWrap='CJK'),
        'cell':ParagraphStyle('cell',fontName='STSong-Light',fontSize=9.3,leading=15,textColor=navy,wordWrap='CJK'),
        'code':ParagraphStyle('code',fontName='Courier',fontSize=8.7,leading=14,textColor=navy,spaceAfter=9)}
    summary=pd.read_csv(ROOT/'results/models/cv_metric_summary.csv').set_index('model')
    stage=json.loads((ROOT/'results/models/stage2_summary.json').read_text())
    eda=json.loads((ROOT/'results/eda_summary.json').read_text())
    story=[]
    def p(text,style='body'):story.append(Paragraph(text.replace('R²','R2'),styles[style]))
    def h(text):p(text,'h')
    def table(rows,widths):
        t=Table([[Paragraph(str(x).replace('R²','R2'),styles['cell']) for x in row] for row in rows],colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9F1F5')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                              ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#D8E0E6'))]))
        story.extend([t,Spacer(1,9)])
    def page():story.append(PageBreak())

    p('基线身体组成与营养的<br/>肌肉厚度预测研究','title')
    p('最终审阅版｜2026 年 10 月 21 日','h')
    p('项目性质：公开纵向抗阻训练数据的二次分析。研究对象是 12 周训练后的股外侧肌厚度，不是疾病风险预测或通用健康预测系统。')
    h('一句话研究问题')
    p('已知训练前的肌肉厚度后，再加入训练前的身体组成、蛋白质和热量摄入，是否能更好地解释和预测 12 周后的肌肉厚度？')
    h('完成状态')
    table([['工作内容','当前状态'],['原始数据核查、清洗与数据字典','已完成'],['描述性分析与图表','已完成'],['回归、Ridge、内部验证与诊断','已完成'],['研究报告与可复现项目包','已完成'],['GitHub 仓库公开发布','尚未发布'],['独立外部数据验证','尚未开展']], [270,205])
    h('核心结论')
    p('在这 67 人的数据与固定验证方案下，加入基线身体组成和营养，没有改善留出预测。这个结论不等于证明营养没有生理作用，也不表示项目未完成。')
    p('本项目使用普通线性回归与 Ridge 正则化回归，没有使用 RNN/LSTM。项目状态可写“已完成数据分析与模型内部验证”，不宜继续笼统标为“进行中”。','small')
    page()

    p('数据与研究方法','title')
    h('数据来源和样本')
    p('来源为 Mobley 等人 2018 年论文的公开 S2 原始 Excel。67 名年轻、此前没有规律抗阻训练的男性，具有 12 周训练前后的配对测量。原始实验和数据采集由该研究团队完成。[1]')
    p('真实参与者每人保留一行；原表的均值、标准差、统计检验和注释行不作为参与者。主要分析字段全部完整，未插补、未删除极值。深蹲力量少 3 人，但不属于主要预测变量，因此主要样本仍为 67 人。')
    table([['角色','实际使用的变量'],['结果变量','12 周后超声测得的股外侧肌厚度（cm）'],['基线调整项','训练前肌肉厚度（cm）'],['四个新增变量','训练前 DXA 瘦体重、脂肪质量、蛋白质 g/day、热量 kcal/day'],['敏感性分析','基线与扩展模型均调整年龄，以及年龄加补充剂分类代码']], [105,370])
    h('怎样比较模型？')
    p('主要比较：只用初始厚度的线性模型、加入四个变量的扩展线性模型，以及使用相同变量的 Ridge 模型；另保留均值参考模型。')
    p('验证采用参与者层面的 20 次重复、每次 5 折划分。所有模型使用相同的外层划分。Ridge 在外层训练数据内部再做 5 折调参，标准化与编码均只用训练数据拟合。每次重复中，每人获得一次由未使用其结果拟合的模型给出的留出预测。')
    p('全样本条件关联采用 HC3 稳健协方差与近似 t/F 参考；原始固定方案在阅读论文后、新模型拟合前形成，并非注册预分析。没有根据模型结果挑选主要变量。')
    h('独立检查')
    p('检查覆盖所有外层与内层划分、唯一留出预测、导出指标重算、基线系数，以及 HC3 的逐人删除独立核验。另用直接公式重建了首个 Ridge 内部搜索与预测。你的 Windows 运行日志复现了主要指标与检查结果。')
    page()

    p('实际结果与含义','title')
    table([['模型','平均绝对误差 MAE','最终厚度 R²','变化量 R²'],
           ['只用初始厚度','0.1861 cm','0.457','0.049'],
           ['增加身体组成与营养','0.1949 cm','0.404','-0.043'],
           ['相同变量使用 Ridge','0.1966 cm','0.403','-0.046']], [180,115,90,90])
    p('MAE 越低越好。0.1861 cm 约为 1.861 mm，表示绝对预测误差的平均量级，不是预测的增肌量。分数先汇总每次重复中的全部留出预测，再对 20 次取均值。重复划分不增加独立参与者人数。','small')
    image=Image(str(ROOT/'figures/04_model_comparison.png'));ratio=image.imageHeight/image.imageWidth
    image.drawWidth=475;image.drawHeight=475*ratio
    story.append(image)
    p('图：左侧为留出预测误差，右侧为观察变化量的 R²。误差线表示不同划分的标准差，不是总体置信区间。','small')
    h('为什么不是营养无效的结论？')
    p('普通扩展模型的平均误差比基线增加约 0.0088 cm，Ridge 也没有改善。四个新增变量的联合 HC3 Wald 近似检验 p=0.589，尚未给出明确的新增条件关联证据；这不是效应为零的证明。')
    h('最终大小与增长量的区别')
    p('训练前肌肉较大的人，训练后仍可能较大。基线模型对最终厚度有一定预测信息，但对“增长多少”的变化 R²只有约 0.049。不能把最终厚度的预测表现宣传成准确的个体增肌预测。')
    page()

    p('研究边界、复现与署名','title')
    h('局限')
    for text in ['样本只有 67 人，且群体单一；没有独立外部数据验证。',
                 '营养是短期自报记录，不是完整训练期的连续饮食。真实体重、身高与睡眠列缺失，未补造相应指标。',
                 '补充剂代码名称及 POST 营养是否包含补充剂尚未确认；没有猜测名称。',
                 '当前分析没有不训练对照或独立重复训练周期；观察到的变化同时包含测量和日常波动。',
                 '系数是条件关联，不是改变蛋白质摄入的因果效应。模型不支持个人增肌处方。']:
        p(text)
    h('在本地复现')
    p('在项目主文件夹打开 PowerShell。没有 .venv 时，先执行 python -m venv .venv。随后依次运行：')
    story.append(Preformatted(r'.\.venv\Scripts\python.exe -m pip install -r requirements.txt',styles['code']))
    story.append(Preformatted(r'.\.venv\Scripts\python.exe src\run_models.py',styles['code']))
    p('完整输出包括清洗数据、图表、模型系数、验证划分、留出预测和独立检查。分析方案日期为 2026-10-13，报告整理日期为 2026-10-21。第三版以后的 CSV 统一为 LF 换行；你提供的原校验值差异可以由 Windows CRLF 换行完全解释。','small')
    h('数据与代码贡献')
    p('原始论文、实验和数据采集：Mobley 等研究团队。新 Python 清洗与建模代码：为本项目使用 AI 辅助生成；未复制原作者分析代码。你已完成本地环境配置与运行复现，其他个人贡献应按实际理解、检查和完成的工作说明。')
    h('参考与项目标注')
    p('[1] Mobley CB et al. (2018). PLOS ONE 13(4):e0195203. S2 File: Raw data. DOI: 10.1371/journal.pone.0195203。原始数据按 CC BY 4.0 署名。','small')
    p('推荐项目名：纵向抗阻训练数据中的增肌反应建模与预测。英文：Muscle Hypertrophy Response Modeling and Prediction。完成状态：数据分析与模型内部验证已完成；GitHub 尚未发布。','small')

    out=ROOT/'reports/Hypertrophy_Research_Report_FINAL_CN.pdf'
    def footer(c,doc):
        c.setFont('STSong-Light',8);c.setFillColor(navy)
        c.drawString(58,30,'增肌反应建模与预测｜最终审阅版｜2026-10-21')
        c.drawRightString(A4[0]-58,30,str(doc.page))
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=58,rightMargin=58,topMargin=48,bottomMargin=58,
                          title='增肌反应建模与预测：最终审阅版',author='公开数据二次分析；AI 辅助代码')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)


if __name__=='__main__':main()
