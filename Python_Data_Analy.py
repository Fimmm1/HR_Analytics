import streamlit as st
st.set_page_config(page_title="HR Attrition Analytics", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
import io
import warnings
warnings.filterwarnings('ignore')

# ─── 폰트 ───
@st.cache_resource
def setup_font():
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)
    for kf in ['NanumGothic','NanumBarunGothic','Malgun Gothic','AppleGothic']:
        if kf in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = kf; break
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
setup_font()

st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub-header{font-size:.95rem;color:#64748b}
div[data-testid="stMetricValue"]{font-size:1.8rem}
.insight-box{background:#f0f4ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0 20px 0;font-size:14px;color:#1e293b}
.warning-box{background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0 20px 0;font-size:14px;color:#1e293b}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 파이프라인
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(eb, sb, tb):
    progress = st.progress(0, text="📂 데이터 로딩 중...")
    emp = pd.read_csv(io.BytesIO(eb), encoding='utf-8-sig')
    survey = pd.read_csv(io.BytesIO(sb))
    training = pd.read_csv(io.BytesIO(tb))
    progress.progress(10, text="✅ 데이터 로드 완료")
    for c in ['DepartmentType','Division','Title','EmployeeStatus','EmployeeType','TerminationType']:
        if c in emp.columns: emp[c]=emp[c].astype(str).str.strip()
    emp.drop(columns=[c for c in ['FirstName','LastName','ADEmail','Supervisor'] if c in emp.columns],inplace=True)
    emp['Attrition']=emp['EmployeeStatus'].apply(lambda x:1 if x in ['Voluntarily Terminated','Terminated for Cause'] else 0)
    emp['StartDate_parsed']=pd.to_datetime(emp['StartDate'],errors='coerce',dayfirst=True)
    emp['ExitDate_parsed']=pd.to_datetime(emp['ExitDate'],errors='coerce',dayfirst=True)
    emp['DOB_parsed']=pd.to_datetime(emp['DOB'],errors='coerce',dayfirst=True)
    ref=emp['ExitDate_parsed'].max(); 
    if pd.isna(ref): ref=pd.Timestamp.now()
    emp['Tenure_Years']=emp.apply(lambda r:(r['ExitDate_parsed']-r['StartDate_parsed']).days/365.25 if pd.notna(r['ExitDate_parsed']) else (ref-r['StartDate_parsed']).days/365.25,axis=1).round(1)
    emp['Age']=((ref-emp['DOB_parsed']).dt.days/365.25).round(0)
    emp['Exit_Year']=emp['ExitDate_parsed'].dt.year
    progress.progress(25,text="✅ 전처리 완료")
    merged=emp.merge(survey,left_on='EmpID',right_on='Employee ID',how='left').merge(training,left_on='EmpID',right_on='Employee ID',how='left')
    merged.drop(columns=['Employee ID_x','Employee ID_y'],inplace=True,errors='ignore')
    progress.progress(35,text="✅ 데이터 통합 완료")
    for c,e in {'DepartmentType':'Dept_enc','Title':'Title_enc','GenderCode':'Gender_enc','EmployeeType':'EmpType_enc','Performance Score':'Perf_enc'}.items():
        merged[e]=LabelEncoder().fit_transform(merged[c].astype(str))
    fc=['Dept_enc','Title_enc','Gender_enc','EmpType_enc','Perf_enc','Current Employee Rating','Engagement Score','Satisfaction Score','Work-Life Balance Score','Training Duration(Days)','Training Cost','LocationCode']
    fl=['Department','Title','Gender','Employee Type','Performance','Employee Rating','Engagement','Satisfaction','Work-Life Balance','Training Duration','Training Cost','Location']
    X=merged[fc].fillna(0);y=merged['Attrition']
    progress.progress(45,text="🔄 SMOTE 처리 중...")
    Xr,yr=SMOTE(random_state=42).fit_resample(X,y)
    Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=0.2,random_state=42,stratify=yr)
    progress.progress(55,text="🧠 Random Forest 학습 중...")
    rf=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=5,random_state=42,n_jobs=-1)
    rf.fit(Xtr,ytr)
    yp=rf.predict(Xte);ypr=rf.predict_proba(Xte)[:,1]
    cm=confusion_matrix(yte,yp);rpt=classification_report(yte,yp,target_names=['Active','Terminated'],output_dict=True);auc=roc_auc_score(yte,ypr)
    fi=pd.DataFrame({'feature':fl,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    progress.progress(65,text="📊 SHAP 분석 중...")
    exp=shap.TreeExplainer(rf)
    Xs=Xte[:300];Xsd=pd.DataFrame(Xs,columns=fl)
    sv=exp.shap_values(Xs)
    if isinstance(sv,list):sc1=sv[1]
    elif sv.ndim==3:sc1=sv[:,:,1]
    else:sc1=sv
    ss=min(800,len(merged));si=np.random.RandomState(42).choice(len(merged),ss,replace=False)
    Xos=merged[fc].fillna(0).iloc[si];Xosd=pd.DataFrame(Xos.values,columns=fl);odepts=merged['DepartmentType'].iloc[si].values
    osv=exp.shap_values(Xos.values)
    if isinstance(osv,list):osc1=osv[1]
    elif osv.ndim==3:osc1=osv[:,:,1]
    else:osc1=osv
    progress.progress(75,text="🏢 조직별 분석 중...")
    dfi={}
    for d in emp['DepartmentType'].unique():
        dm=merged['DepartmentType']==d;Xd=merged.loc[dm,fc].fillna(0);yd=merged.loc[dm,'Attrition']
        if yd.sum()>=15 and len(yd)>=50:
            try:
                sm=SMOTE(random_state=42,k_neighbors=min(5,int(yd.sum())-1));Xdr,ydr=sm.fit_resample(Xd,yd)
                rfd=RandomForestClassifier(n_estimators=80,max_depth=8,random_state=42,n_jobs=-1);rfd.fit(Xdr,ydr)
                dfi[d]=pd.DataFrame({'feature':fl,'importance':rfd.feature_importances_}).sort_values('importance',ascending=False)
            except:pass
    dsv={}
    for d in emp['DepartmentType'].unique():
        sub=merged[merged['DepartmentType']==d]
        if sub['Attrition'].sum()>=5:
            sv2=sub.groupby('Attrition')[['Engagement Score','Satisfaction Score','Work-Life Balance Score']].mean().round(2)
            if 0 in sv2.index and 1 in sv2.index:dsv[d]=sv2
    progress.progress(85,text="📋 직원 스코어링 중...")
    Xa=merged[fc].fillna(0);merged['Risk_Score']=(rf.predict_proba(Xa)[:,1]*100).round(1)
    merged['Risk_Level']=merged['Risk_Score'].apply(lambda x:'🔴 High' if x>=60 else '🟡 Medium' if x>=30 else '🟢 Low')
    ds=emp.groupby('DepartmentType').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index();ds['rate']=(ds['terminated']/ds['total']*100).round(1)
    ts=emp.groupby('Title').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index();ts['rate']=(ts['terminated']/ts['total']*100).round(1)
    cs=emp.groupby(['DepartmentType','Title']).agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index();cs['rate']=(cs['terminated']/cs['total']*100).round(1)
    exited=emp[(emp['Attrition']==1)&(emp['ExitDate_parsed'].notna())]
    ye=exited.groupby('Exit_Year')['EmpID'].count().reset_index();ye.columns=['year','exits']
    yd2=[]
    for _,r in ye.iterrows():
        yr=int(r['year']);act=len(emp[(emp['StartDate_parsed'].dt.year<=yr)&((emp['ExitDate_parsed'].isna())|(emp['ExitDate_parsed'].dt.year>=yr))])
        yd2.append({'year':yr,'exits':int(r['exits']),'active':act,'rate':round(r['exits']/max(act,1)*100,1)})
    ydf=pd.DataFrame(yd2)
    scomp=merged.groupby('Attrition')[['Engagement Score','Satisfaction Score','Work-Life Balance Score']].mean().round(2)
    progress.progress(100,text="✅ 분석 완료!")
    return {'emp':emp,'merged':merged,'exited':exited,'rf':rf,'fi':fi,'cm':cm,'rpt':rpt,'auc':auc,
            'sc1':sc1,'Xsd':Xsd,'osc1':osc1,'Xosd':Xosd,'odepts':odepts,
            'ds':ds,'ts':ts,'cs':cs,'ydf':ydf,'scomp':scomp,'fl':fl,'dfi':dfi,'dsv':dsv}

# ═══════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════
def gen_pdf(R,sd):
    from fpdf import FPDF;import tempfile,os
    emp=R['emp'];total=len(emp);termed=int(emp['Attrition'].sum());rate=round(termed/total*100,1)
    avg_rate=R['ds']['rate'].mean()
    high_depts=R['ds'][R['ds']['rate']>=15].sort_values('rate',ascending=False)
    fi_src=R['dfi'].get(sd,R['fi']) if sd!='All' else R['fi']
    top3fi=fi_src.head(3)

    def svc(fig,n):
        p=os.path.join(tempfile.gettempdir(),f'{n}.png');fig.savefig(p,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig);return p

    # 차트 1: 조직별 이탈률
    dd=R['ds'].sort_values('rate',ascending=True)
    f1,ax=plt.subplots(figsize=(10,5));colors=[('#ef4444' if r>=15 else '#f59e0b' if r>=5 else '#22c55e') for r in dd['rate']]
    ax.barh(dd['DepartmentType'],dd['rate'],color=colors,height=0.5)
    for i,(r,t) in enumerate(zip(dd['rate'],dd['total'])):ax.text(r+0.3,i,f'{r}% ({t})',va='center',fontsize=9)
    ax.set_xlabel('Attrition Rate (%)');ax.set_title('Department Attrition Rate',fontweight='bold');plt.tight_layout();c1=svc(f1,'d')

    # 차트 2: Feature Importance
    fiv=fi_src.sort_values('importance',ascending=True)
    f2,ax=plt.subplots(figsize=(10,6));ax.barh(fiv['feature'],fiv['importance'],color=plt.cm.viridis(np.linspace(0.3,0.9,len(fiv))),height=0.5)
    ax.set_title('Feature Importance (Random Forest)',fontweight='bold');plt.tight_layout();c2=svc(f2,'f')

    # 차트 3: 연도별 추이
    f3,ax=plt.subplots(figsize=(10,5));ax.bar(R['ydf']['year'],R['ydf']['exits'],color='#ef4444',alpha=0.7);ax.plot(R['ydf']['year'],R['ydf']['exits'],'o-',color='#991b1b',lw=2)
    for x,y in zip(R['ydf']['year'],R['ydf']['exits']):ax.text(x,y+3,str(y),ha='center',fontweight='bold')
    ax.set_title('Yearly Exit Count',fontweight='bold');plt.tight_layout();c3=svc(f3,'y')

    # 차트 4: 연도별 이탈률
    f3b,ax=plt.subplots(figsize=(10,5));ax.plot(R['ydf']['year'],R['ydf']['rate'],'o-',color='#ef4444',lw=2,markersize=8,markerfacecolor='white',markeredgewidth=2)
    ax.fill_between(R['ydf']['year'],R['ydf']['rate'],alpha=0.1,color='#ef4444')
    for x,y in zip(R['ydf']['year'],R['ydf']['rate']):ax.text(x,y+0.3,f'{y}%',ha='center',fontweight='bold')
    ax.set_title('Yearly Attrition Rate Trend',fontweight='bold');plt.tight_layout();c3b=svc(f3b,'yr')

    # 차트 5: 위험도 분포
    m=R['merged'] if sd=='All' else R['merged'][R['merged']['DepartmentType']==sd]
    f4,ax=plt.subplots(figsize=(10,5));ax.hist(m['Risk_Score'],bins=20,color='#3b82f6',alpha=0.7,edgecolor='white')
    ax.axvline(60,color='#ef4444',ls='--',lw=2,label='High(60)');ax.axvline(30,color='#f59e0b',ls='--',lw=2,label='Mid(30)')
    ax.set_title('Risk Score Distribution',fontweight='bold');ax.legend();plt.tight_layout();c4=svc(f4,'r')

    # 차트 6: 서베이 비교
    sc=R['scomp'];sc.index=['Active','Terminated']
    f5,ax=plt.subplots(figsize=(10,5));x=np.arange(3);w=0.3
    ax.bar(x-w/2,sc.iloc[0],w,label='Active',color='#3b82f6');ax.bar(x+w/2,sc.iloc[1],w,label='Terminated',color='#ef4444')
    ax.set_xticks(x);ax.set_xticklabels(['Engagement','Satisfaction','Work-Life Balance'])
    ax.set_ylabel('Score');ax.legend();ax.set_ylim(2.0,4.0);ax.set_title('Survey: Active vs Terminated',fontweight='bold')
    plt.tight_layout();c5=svc(f5,'sv')

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica','B',9);self.set_text_color(120,120,120)
            self.cell(0,8,'HR Attrition Analysis Report  |  Confidential',align='R',new_x="LMARGIN",new_y="NEXT")
            self.set_draw_color(200,200,200);self.line(10,self.get_y(),200,self.get_y());self.ln(3)
        def footer(self):
            self.set_y(-15);self.set_font('Helvetica','I',8);self.set_text_color(150,150,150)
            self.cell(0,10,f'Page {self.page_no()}  |  Generated by HR Attrition Analytics',align='C')
        def s1(self,t):self.set_font('Helvetica','B',14);self.set_text_color(25,25,25);self.cell(0,10,t,new_x="LMARGIN",new_y="NEXT");self.ln(2)
        def s2(self,t):self.set_font('Helvetica','B',11);self.set_text_color(50,50,50);self.cell(0,7,t,new_x="LMARGIN",new_y="NEXT");self.ln(1)
        def bd(self,t):self.set_font('Helvetica','',9);self.set_text_color(50,50,50);self.multi_cell(0,5,t);self.ln(2)
        def insight(self,t):
            self.set_fill_color(240,244,255);self.set_draw_color(59,130,246)
            self.set_font('Helvetica','',9);self.set_text_color(30,30,30)
            x=self.get_x();y=self.get_y();self.rect(x,y,190,self.get_string_width(t)/180*5+12)
            self.set_xy(x+2,y+2);self.multi_cell(186,5,f'[Insight] {t}');self.ln(4)
        def warning(self,t):
            self.set_fill_color(254,243,199);self.set_draw_color(245,158,11)
            self.set_font('Helvetica','B',9);self.set_text_color(120,60,0)
            self.multi_cell(0,5,f'[Warning] {t}');self.ln(3)

    pdf=PDF();pdf.set_auto_page_break(auto=True,margin=20)

    # ─── 표지 ───
    pdf.add_page();pdf.ln(50)
    pdf.set_font('Helvetica','B',32);pdf.set_text_color(25,25,25)
    pdf.cell(0,15,'HR Attrition',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,15,'Analysis Report',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.ln(8);pdf.set_draw_color(59,130,246);pdf.set_line_width(0.8);pdf.line(70,pdf.get_y(),140,pdf.get_y());pdf.ln(8)
    pdf.set_font('Helvetica','',12);pdf.set_text_color(100,100,100)
    scope=sd if sd!='All' else 'All Departments'
    pdf.cell(0,7,f'Analysis Scope: {scope}',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,7,f'Total Employees: {total:,}  |  Attrition Rate: {rate}%',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,7,f'Model Performance: ROC-AUC {R["auc"]:.4f}',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.ln(30)
    pdf.set_font('Helvetica','',10)
    pdf.cell(0,7,'Sogang University AI/SW Graduate School',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,7,'Kim Hyuntae (A74032)',align='C',new_x="LMARGIN",new_y="NEXT")

    # ─── 1. Executive Summary ───
    pdf.add_page();pdf.s1('1. Executive Summary')
    pdf.bd(f'This report presents the results of an employee attrition analysis conducted on {total:,} employee records. Using Random Forest machine learning and SHAP explainability framework, we identified key attrition drivers, high-risk departments, and individual employee risk scores to enable proactive retention strategies.')
    pdf.s2('Key Performance Indicators')
    pdf.set_font('Helvetica','B',9)
    for h in ['Metric','Value','Assessment']:pdf.cell(63,7,h,border=1,align='C')
    pdf.ln();pdf.set_font('Helvetica','',9)
    for met,val,ass in [('Total Employees',f'{total:,}','-'),('Terminated',f'{termed}','-'),('Attrition Rate',f'{rate}%','CRITICAL' if rate>=15 else 'WARNING' if rate>=10 else 'NORMAL'),('ROC-AUC Score',f'{R["auc"]:.4f}','Excellent' if R["auc"]>=0.85 else 'Good' if R["auc"]>=0.75 else 'Fair'),('High-Risk Employees',f'{len(m[m["Risk_Score"]>=60])}','Immediate attention needed')]:
        pdf.cell(63,6,met,border=1);pdf.cell(63,6,val,border=1,align='C');pdf.cell(63,6,ass,border=1,align='C');pdf.ln()
    pdf.ln(3)
    if len(high_depts)>0:
        pdf.warning(f'High-risk departments identified: {", ".join([f"{r.DepartmentType}({r.rate}%)" for _,r in high_depts.iterrows()])}. These departments significantly exceed the company average of {avg_rate:.1f}% and require immediate retention intervention.')
    pdf.s2('Top 3 Attrition Drivers')
    pdf.bd(f'The Random Forest model identified the following as the most influential factors in predicting employee attrition:\n  1. {top3fi.iloc[0]["feature"]} (Importance: {top3fi.iloc[0]["importance"]:.1%})\n  2. {top3fi.iloc[1]["feature"]} (Importance: {top3fi.iloc[1]["importance"]:.1%})\n  3. {top3fi.iloc[2]["feature"]} (Importance: {top3fi.iloc[2]["importance"]:.1%})\nThese three factors account for {top3fi["importance"].sum():.1%} of the model\'s predictive power.')

    # ─── 2. Department Analysis ───
    pdf.add_page();pdf.s1('2. Department Attrition Analysis')
    pdf.bd('The following chart and table show attrition rates by department. Departments are color-coded: Red (>=15%, Critical), Yellow (5-15%, Warning), Green (<5%, Stable).')
    pdf.image(c1,x=10,w=190);pdf.ln(3)
    pdf.set_font('Helvetica','B',9)
    for h in ['Department','Total','Terminated','Rate (%)','Risk Level']:pdf.cell(38,7,h,border=1,align='C')
    pdf.ln();pdf.set_font('Helvetica','',9)
    for _,r in R['ds'].sort_values('rate',ascending=False).iterrows():
        lv='CRITICAL' if r['rate']>=15 else 'WARNING' if r['rate']>=5 else 'STABLE'
        pdf.cell(38,6,str(r['DepartmentType'])[:18],border=1);pdf.cell(38,6,str(r['total']),border=1,align='C')
        pdf.cell(38,6,str(int(r['terminated'])),border=1,align='C');pdf.cell(38,6,str(r['rate']),border=1,align='C');pdf.cell(38,6,lv,border=1,align='C');pdf.ln()
    pdf.ln(3)
    low_depts=R['ds'][R['ds']['rate']<5]['DepartmentType'].tolist()
    if high_depts is not None and len(high_depts)>0:
        pdf.bd(f'[Analysis] {", ".join(high_depts["DepartmentType"].tolist())} show attrition rates significantly above company average ({avg_rate:.1f}%). Immediate investigation into root causes is recommended, with focus on compensation competitiveness, workload distribution, and career development opportunities.')
    if low_depts:
        pdf.bd(f'[Best Practice] {", ".join(low_depts)} maintain attrition below 5%. Success factors from these departments should be analyzed and benchmarked for cross-departmental application.')

    # ─── 3. Attrition Drivers ───
    pdf.add_page();pdf.s1('3. Key Attrition Drivers (Feature Importance)')
    pdf.bd('Feature Importance measures how much each variable contributes to the model\'s ability to predict attrition. Higher values indicate stronger predictive power. Note: this shows WHICH factors matter, not HOW they affect attrition (see SHAP analysis for directionality).')
    pdf.image(c2,x=10,w=190);pdf.ln(3)
    pdf.s2('Top 5 Features - Detailed Interpretation')
    pdf.set_font('Helvetica','B',8)
    pdf.cell(10,7,'#',border=1,align='C');pdf.cell(45,7,'Feature',border=1);pdf.cell(25,7,'Score',border=1,align='C');pdf.cell(110,7,'Business Interpretation',border=1);pdf.ln()
    pdf.set_font('Helvetica','',8)
    interps={'Training Cost':'Training investment level strongly correlates with retention. Under-invested employees may feel undervalued.','Location':'Geographic factors create significant attrition variance. Certain sites may have local market or commute issues.','Title':'Specific job roles show inherently higher turnover, reflecting market demand and career path limitations.','Engagement':'Lower engagement scores serve as leading indicators of impending departure.','Satisfaction':'Job satisfaction directly impacts the decision to stay or leave the organization.','Work-Life Balance':'Poor work-life balance is a key push factor driving employees to seek alternatives.','Employee Rating':'Performance evaluation outcomes influence career trajectory decisions.','Training Duration':'Duration of training programs reflects development investment depth.','Department':'Organizational culture and management practices vary by department.','Performance':'Performance assessment results affect both voluntary and involuntary turnover.','Employee Type':'Employment classification (FT/PT/Contract) affects job stability expectations.','Gender':'Gender-based patterns may indicate equity issues requiring investigation.'}
    for i,(_,r) in enumerate(fi_src.head(5).iterrows()):
        pdf.cell(10,12,str(i+1),border=1,align='C');pdf.cell(45,12,str(r['feature']),border=1)
        pdf.cell(25,12,f'{r["importance"]:.4f}',border=1,align='C')
        interp=interps.get(r['feature'],'Contributing factor to attrition prediction.')
        # Handle long text
        pdf.cell(110,12,interp[:55],border=1);pdf.ln()
    pdf.ln(3)
    pdf.bd(f'[Key Finding] The top 3 features ({", ".join(top3fi["feature"].tolist())}) account for {top3fi["importance"].sum():.1%} of total predictive power. HR interventions should prioritize these areas for maximum retention impact.')

    # ─── 4. Yearly Trend ───
    pdf.add_page();pdf.s1('4. Yearly Attrition Trend')
    pdf.bd('Understanding temporal patterns helps identify whether attrition is a structural issue or linked to specific organizational events.')
    pdf.s2('4-1. Annual Exit Count')
    pdf.image(c3,x=10,w=190);pdf.ln(3)
    pdf.s2('4-2. Annual Attrition Rate')
    pdf.image(c3b,x=10,w=190);pdf.ln(3)
    pdf.set_font('Helvetica','B',9)
    for h in ['Year','Exits','Est. Active','Rate (%)']:pdf.cell(47,7,h,border=1,align='C')
    pdf.ln();pdf.set_font('Helvetica','',9)
    for _,r in R['ydf'].iterrows():
        pdf.cell(47,6,str(int(r['year'])),border=1,align='C');pdf.cell(47,6,str(r['exits']),border=1,align='C')
        pdf.cell(47,6,str(r['active']),border=1,align='C');pdf.cell(47,6,str(r['rate']),border=1,align='C');pdf.ln()
    pdf.ln(3)
    if len(R['ydf'])>=2:
        first=R['ydf'].iloc[0];last=R['ydf'].iloc[-1]
        trend='increasing' if last['rate']>first['rate'] else 'decreasing' if last['rate']<first['rate'] else 'stable'
        pdf.bd(f'[Trend Analysis] Attrition has been {trend} from {first["rate"]}% ({int(first["year"])}) to {last["rate"]}% ({int(last["year"])}). {"An increasing trend warrants investigation into organizational changes, policy shifts, or market conditions during this period." if trend=="increasing" else "The current trajectory should be maintained through continued application of existing retention strategies."}')

    # ─── 5. Survey Analysis ───
    pdf.add_page();pdf.s1('5. Employee Survey Analysis')
    pdf.bd('Comparing survey scores between active and terminated employees reveals which engagement dimensions serve as leading indicators of attrition risk.')
    pdf.image(c5,x=10,w=190);pdf.ln(3)
    pdf.set_font('Helvetica','B',9)
    for h in ['Survey Item','Active Avg','Terminated Avg','Difference','Signal']:pdf.cell(38,7,h,border=1,align='C')
    pdf.ln();pdf.set_font('Helvetica','',9)
    cols=['Engagement Score','Satisfaction Score','Work-Life Balance Score']
    short_names=['Engagement','Satisfaction','Work-Life Bal.']
    for col,sn in zip(cols,short_names):
        act_v=sc.loc['Active',col];term_v=sc.loc['Terminated',col];diff=term_v-act_v
        signal='WARNING' if diff<-0.05 else 'MONITOR' if diff<0 else 'OK'
        pdf.cell(38,6,sn,border=1);pdf.cell(38,6,f'{act_v:.2f}',border=1,align='C')
        pdf.cell(38,6,f'{term_v:.2f}',border=1,align='C');pdf.cell(38,6,f'{diff:+.2f}',border=1,align='C');pdf.cell(38,6,signal,border=1,align='C');pdf.ln()
    pdf.ln(3)
    neg_items=[sn for col,sn in zip(cols,short_names) if sc.loc['Terminated',col]<sc.loc['Active',col]]
    if neg_items:
        pdf.bd(f'[Survey Insight] Terminated employees scored lower on: {", ".join(neg_items)}. These items function as early warning signals. HR should implement regular pulse surveys and flag employees/teams showing declining scores in these areas for proactive intervention.')

    # ─── 6. Risk Scoring ───
    pdf.add_page();pdf.s1('6. Employee Risk Scoring')
    pdf.bd(f'Each employee was assigned an attrition risk score (0-100) based on Random Forest prediction probability. Scores of 60+ indicate high risk requiring immediate attention, 30-59 indicate medium risk for monitoring, and below 30 indicate low risk.')
    pdf.image(c4,x=10,w=190);pdf.ln(3)
    hi=len(m[m['Risk_Score']>=60]);mi=len(m[(m['Risk_Score']>=30)&(m['Risk_Score']<60)]);lo=len(m[m['Risk_Score']<30])
    pdf.s2('Risk Distribution Summary')
    pdf.set_font('Helvetica','B',9)
    for h in ['Risk Level','Criteria','Count','Percentage','Action Required']:pdf.cell(38,7,h,border=1,align='C')
    pdf.ln();pdf.set_font('Helvetica','',9)
    ttl=max(len(m),1)
    for lv,cr,cnt,act in [('HIGH','Score >= 60',hi,'Immediate 1:1 interview'),('MEDIUM','Score 30-59',mi,'Regular monitoring'),('LOW','Score < 30',lo,'Standard management')]:
        pdf.cell(38,6,lv,border=1,align='C');pdf.cell(38,6,cr,border=1,align='C');pdf.cell(38,6,str(cnt),border=1,align='C')
        pdf.cell(38,6,f'{cnt/ttl*100:.1f}%',border=1,align='C');pdf.cell(38,6,act,border=1,align='C');pdf.ln()
    pdf.ln(3)
    pdf.s2('Top 20 High-Risk Employees')
    pdf.set_font('Helvetica','B',7)
    for h in ['ID','Department','Title','Risk%','Tenure','Performance','Rating']:
        w=27 if h in['Department','Title'] else 22 if h=='Performance' else 17
        pdf.cell(w,6,h,border=1,align='C')
    pdf.ln();pdf.set_font('Helvetica','',7)
    for _,r in m.nlargest(20,'Risk_Score').iterrows():
        pdf.cell(17,5,str(r['EmpID']),border=1,align='C');pdf.cell(27,5,str(r['DepartmentType'])[:14],border=1)
        pdf.cell(27,5,str(r['Title'])[:14],border=1);pdf.cell(17,5,str(r['Risk_Score']),border=1,align='C')
        pdf.cell(17,5,str(r.get('Tenure_Years','N/A')),border=1,align='C')
        pdf.cell(22,5,str(r.get('Performance Score','N/A'))[:10],border=1,align='C')
        pdf.cell(17,5,str(r.get('Current Employee Rating','N/A')),border=1,align='C');pdf.ln()
    pdf.ln(3)
    if hi>0:
        pdf.bd(f'[Action Required] {hi} employees are classified as high-risk (score >= 60). Recommended immediate actions: (1) Schedule 1:1 stay interviews within 2 weeks, (2) Review compensation competitiveness for each individual, (3) Assess workload and career development satisfaction, (4) Develop personalized retention plans.')

    # ─── 7. Recommendations ───
    pdf.add_page();pdf.s1('7. Strategic Recommendations')
    pdf.bd('Based on the comprehensive analysis above, the following strategic recommendations are organized by department risk level and priority.')
    for _,r in R['ds'].sort_values('rate',ascending=False).iterrows():
        if sd!='All' and r['DepartmentType']!=sd:continue
        rt=r['rate'];pdf.s2(f'{r["DepartmentType"]} (Attrition: {rt}%)')
        if rt>=15:
            pdf.bd(f'Risk Level: CRITICAL\n\nShort-term Actions (0-3 months):\n  1. Deploy emergency retention packages for top {min(int(r["terminated"]),20)} high-risk employees\n  2. Conduct stay interviews with all team members within 30 days\n  3. Review and adjust compensation to match market rates\n\nMid-term Actions (3-12 months):\n  4. Establish clear career progression pathways (e.g., Technician I > II > Manager)\n  5. Implement flexible work arrangements and workload optimization\n  6. Launch mentoring program pairing senior and junior staff\n\nTarget: Reduce attrition from {rt}% to {max(rt-5,5):.0f}% within 12 months\nEstimated savings: {int(r["terminated"])*0.5:.0f} fewer departures = significant recruitment cost reduction')
        elif rt>=5:
            pdf.bd(f'Risk Level: WARNING\n\nActions:\n  1. Identify and address specific high-turnover job roles within the department\n  2. Strengthen internal mobility and job rotation programs\n  3. Enhance regular engagement surveys with follow-up action plans\n\nTarget: Maintain attrition below {max(rt-2,3):.0f}% within 12 months')
        else:
            pdf.bd(f'Risk Level: STABLE\n\nActions:\n  1. Maintain current HR policies and monitoring cadence\n  2. Document and share retention success factors for cross-department benchmarking\n  3. Continue periodic pulse surveys to detect early warning signs')

    # ─── 8. Methodology ───
    pdf.add_page();pdf.s1('8. Methodology & Limitations')
    pdf.s2('Analysis Pipeline')
    pdf.bd(f'1. Data Integration: 3 CSV tables joined on Employee ID ({total:,} records)\n2. Preprocessing: String normalization, binary encoding, derived features (tenure, age)\n3. Class Imbalance: SMOTE oversampling applied (minority class {termed} -> balanced)\n4. Model: Random Forest Classifier (100 trees, max_depth=10)\n5. Evaluation: ROC-AUC {R["auc"]:.4f}, train/test split 80/20 with stratification\n6. Explainability: SHAP TreeExplainer for feature contribution analysis\n7. Risk Scoring: Model prediction probability scaled to 0-100 for each employee')
    pdf.s2('Limitations & Considerations')
    pdf.bd('1. This analysis is based on historical data and statistical patterns. Individual circumstances not captured in the data may influence actual attrition decisions.\n2. SMOTE-generated synthetic samples may introduce slight bias in model evaluation metrics.\n3. Feature Importance shows correlation, not causation. Organizational context is needed for causal interpretation.\n4. Risk scores should be used as decision-support tools, not as deterministic predictions.\n5. Regular model retraining with updated data is recommended to maintain prediction accuracy.')

    # ─── Disclaimer ───
    pdf.ln(10);pdf.set_font('Helvetica','I',8);pdf.set_text_color(130,130,130)
    pdf.multi_cell(0,4,'DISCLAIMER: This report is generated by an AI-powered HR analytics system and is intended as a decision-support tool. All recommendations should be reviewed by HR professionals considering organizational context, labor regulations, and individual employee circumstances before implementation. Employee data privacy must be maintained in accordance with applicable data protection regulations.')

    for f in [c1,c2,c3,c3b,c4,c5]:
        try:os.remove(f)
        except:pass
    return bytes(pdf.output())

# ═══════════════════════════════════════════════════════════════
# ChatGPT
# ═══════════════════════════════════════════════════════════════
def get_ai_plan(key,dept,rate,total,fi_text,ctx):
    try:
        from openai import OpenAI;client=OpenAI(api_key="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")
        prompt=f"""당신은 글로벌 HR 컨설팅 펌의 시니어 HR 전략 컨설턴트입니다.
반드시 아래 데이터 수치를 근거로만 답변하세요.
[분석 결과] {ctx}
[대상] {dept}: {total}명, 이탈률 {rate}%
[Feature Importance] {fi_text}
## 📊 {dept} 이탈 분석 리포트
### 1. 현황 진단 (전사 평균 비교, 영향도 정량화)
### 2. 핵심 이탈 원인 3가지 ([근거: 수치] 포함)
### 3. 단기 액션 (0~3개월) - 표: 우선순위|시책|대상|방법|KPI
### 4. 중장기 액션 (3~12개월) - 표 형태
### 5. 기대 효과 (이탈률 목표, 비용 절감)
### ⚠️ 본 분석은 의사결정 참고자료이며 최종 판단은 HR 담당자 검토 필요"""
        resp=client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"system","content":"15년 경력 글로벌 HR 전략 컨설턴트. 한국어 답변."},{"role":"user","content":prompt}],max_tokens=2000,temperature=0.3)
        return resp.choices[0].message.content
    except Exception as e:return f"API Error: {e}"

# ═══════════════════════════════════════════════════════════════
# 차트 + 설명 헬퍼
# ═══════════════════════════════════════════════════════════════
def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>',unsafe_allow_html=True)
def warn_insight(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown('<h1 class="main-header">🏢 HR Attrition Analytics</h1>',unsafe_allow_html=True)
    st.markdown('<p class="sub-header">조직·직무별 이탈 패턴 분석 및 HR 액션 플랜 자동화 · AI·SW대학원 김현태 / A74032</p>',unsafe_allow_html=True)
    st.divider()
    if 'results' not in st.session_state:st.session_state.results=None
    with st.sidebar:
        st.header("📂 데이터 업로드")
        st.caption("CSV 파일을 드래그하거나 Browse files로 업로드")
        ef=st.file_uploader("① 직원 인사정보",type=['csv'],help="employee_data.csv")
        sf=st.file_uploader("② 직원 설문조사",type=['csv'],help="engagement_survey.csv")
        tf=st.file_uploader("③ 직원 교육정보",type=['csv'],help="training_data.csv")
        if ef and sf and tf:
            st.success("✅ 3개 파일 업로드 완료")
            if st.button("🧠 AI 이탈 분석 시작",type="primary",use_container_width=True):
                st.session_state.results=run_pipeline(ef.getvalue(),sf.getvalue(),tf.getvalue());st.rerun()
        else:st.info(f"📎 {sum([1 for f in[ef,sf,tf]if f])}/3")
        if st.session_state.results:
            st.divider();st.header("🔍 필터");R=st.session_state.results
            sd=st.selectbox("Department",['전체']+sorted(R['ds']['DepartmentType'].tolist()),key='sel_dept')
            if sd=='전체':tl=['전체']+sorted(R['ts']['Title'].tolist())
            else:tl=['전체']+sorted(R['cs'][R['cs']['DepartmentType']==sd]['Title'].unique().tolist())
            st.selectbox("Job Title",tl,key='sel_title')
            st.divider();st.header("⚙️ AI 설정")
            st.text_input("OpenAI API Key",type="password",placeholder="sk-...",key='api_key',value=st.secrets.get("OPENAI_API_KEY","") if hasattr(st,'secrets') else "")

    if st.session_state.results is None:
        st.markdown("### 👋 시작하기")
        st.markdown("왼쪽 사이드바에서 **3개의 CSV 파일**을 업로드한 후 **AI 이탈 분석 시작** 버튼을 클릭하세요.")
        c1,c2,c3=st.columns(3)
        c1.info("📁 **employee_data.csv**\n\n직원 인사정보 (26개 변수)")
        c2.info("📁 **engagement_survey.csv**\n\n직원 설문조사 (5개 변수)")
        c3.info("📁 **training_data.csv**\n\n교육 훈련 정보 (9개 변수)")
        return

    R=st.session_state.results;emp=R['emp']
    sd=st.session_state.get('sel_dept','전체');stl=st.session_state.get('sel_title','전체')
    ak=st.session_state.get('api_key','')
    filt=emp.copy()
    if sd!='전체':filt=filt[filt['DepartmentType']==sd]
    if stl!='전체':filt=filt[filt['Title']==stl]
    kt=len(filt);kterm=int(filt['Attrition'].sum());kr=round(kterm/max(kt,1)*100,1)

    k1,k2,k3,k4=st.columns(4)
    k1.metric("전체 인원",f"{kt:,}명");k2.metric("퇴직자 수",f"{kterm}명")
    k3.metric("이탈률",f"{kr}%",delta="위험" if kr>=15 else "주의" if kr>=10 else "양호",delta_color="inverse" if kr>=10 else "normal")
    k4.metric("재직자",f"{kt-kterm:,}명")
    st.divider()

    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊 이탈 분석","🧠 주요 원인 분석","📈 모델 성능","👤 직원 스코어링","🎯 HR 액션 플랜","📥 보고서"])

    # ═══ TAB 1 ═══
    with tab1:
        # 조직별 이탈률
        st.subheader("📉 조직별 이탈률")
        dd=R['ds'].sort_values('rate',ascending=True)
        fig,ax=plt.subplots(figsize=(14,6))
        colors=[('#ef4444' if r>=15 else '#f59e0b' if r>=5 else '#22c55e') for r in dd['rate']]
        ax.barh(dd['DepartmentType'],dd['rate'],color=colors,height=0.5)
        for i,(r,t) in enumerate(zip(dd['rate'],dd['total'])):ax.text(r+0.3,i,f'{r}%  ({t}명)',va='center',fontsize=11,fontweight='bold')
        ax.set_xlabel('Attrition Rate (%)',fontsize=12);ax.set_title('Department Attrition Rate',fontsize=16,fontweight='bold');ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        # 고위험/저위험 조직 분석
        high_depts=R['ds'][R['ds']['rate']>=15]['DepartmentType'].tolist()
        low_depts=R['ds'][R['ds']['rate']<5]['DepartmentType'].tolist()
        if high_depts:
            warn_insight(f"**고위험 조직:** {', '.join(high_depts)} — 전사 평균({R['ds']['rate'].mean():.1f}%)을 크게 상회하며, 즉시 리텐션 대응이 필요합니다.")
        if low_depts:
            insight(f"**안정 조직:** {', '.join(low_depts)} — 이탈률이 5% 미만으로 양호합니다. 해당 조직의 리텐션 성공 요인을 분석하여 고위험 조직에 벤치마킹할 수 있습니다.")

        st.markdown("---")

        # 직무별 이탈률
        st.subheader("📊 직무별 이탈률 Top 10")
        td=R['ts'].copy()
        if sd!='전체':td=R['cs'][R['cs']['DepartmentType']==sd].copy()
        top10=td.nlargest(10,'rate').sort_values('rate',ascending=True)
        fig,ax=plt.subplots(figsize=(14,7))
        colors=[('#ef4444' if r>=20 else '#f59e0b' if r>=10 else '#22c55e') for r in top10['rate']]
        ax.barh(top10['Title'],top10['rate'],color=colors,height=0.5)
        for i,(r,t) in enumerate(zip(top10['rate'],top10['total'])):ax.text(r+0.3,i,f'{r}%  (n={t})',va='center',fontsize=11,fontweight='bold')
        ax.set_xlabel('Attrition Rate (%)',fontsize=12);ax.set_title('Title Attrition Rate Top 10',fontsize=16,fontweight='bold');ax.tick_params(labelsize=10)
        plt.tight_layout();st.pyplot(fig);plt.close()
        top1=td.nlargest(1,'rate').iloc[0] if len(td)>0 else None
        if top1 is not None and top1['rate']>=15:
            warn_insight(f"**최고 위험 직무:** {top1['Title']} ({top1['rate']}%, {int(top1['total'])}명 중 {int(top1['terminated'])}명 퇴직) — 해당 직무의 퇴직 사유와 보상 경쟁력을 집중 점검해야 합니다.")

        st.markdown("---")

        # 연도별 퇴직 건수
        st.subheader("📅 연도별 퇴직 건수")
        fig,ax=plt.subplots(figsize=(14,5))
        ax.bar(R['ydf']['year'],R['ydf']['exits'],color='#ef4444',alpha=0.7,width=0.6)
        ax.plot(R['ydf']['year'],R['ydf']['exits'],'o-',color='#991b1b',lw=2.5,markersize=8)
        for x,y in zip(R['ydf']['year'],R['ydf']['exits']):ax.text(x,y+5,str(y),ha='center',fontweight='bold',fontsize=12)
        ax.set_title('Yearly Exit Count',fontsize=16,fontweight='bold');ax.set_xlabel('Year',fontsize=12);ax.set_ylabel('Exits',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        if len(R['ydf'])>=2:
            first=R['ydf'].iloc[0];last=R['ydf'].iloc[-1]
            insight(f"**연도별 추이:** {int(first['year'])}년 {first['exits']}건 → {int(last['year'])}년 {last['exits']}건으로 퇴직 건수가 변화하고 있습니다. 추세가 증가하고 있다면 조직 내부 변화(구조조정, 정책 변경 등)와의 연관성을 확인해야 합니다.")

        st.markdown("---")

        # 연도별 이탈률 추이
        st.subheader("📅 연도별 이탈률 추이")
        fig,ax=plt.subplots(figsize=(14,5))
        ax.plot(R['ydf']['year'],R['ydf']['rate'],'o-',color='#ef4444',lw=2.5,markersize=10,markerfacecolor='white',markeredgewidth=2.5)
        ax.fill_between(R['ydf']['year'],R['ydf']['rate'],alpha=0.1,color='#ef4444')
        for x,y in zip(R['ydf']['year'],R['ydf']['rate']):ax.text(x,y+0.5,f'{y}%',ha='center',fontweight='bold',fontsize=12)
        ax.set_title('Yearly Attrition Rate',fontsize=16,fontweight='bold');ax.set_xlabel('Year',fontsize=12);ax.set_ylabel('Rate (%)',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        insight(f"**이탈률 추이:** 연도별 이탈률의 증감 추세를 모니터링하여, 급등 시점에 어떤 조직 변화가 있었는지 역추적하는 것이 중요합니다.")

    # ═══ TAB 2 ═══
    with tab2:
        if sd!='전체':st.info(f"📌 **{sd}** 조직 분석 결과입니다.")
        else:st.info("📌 **전체** 데이터 기반 분석 결과입니다. 사이드바에서 조직을 선택하면 해당 조직 분석으로 전환됩니다.")

        # Feature Importance
        if sd!='전체' and sd in R['dfi']:
            st.subheader(f"🧠 Feature Importance — {sd}")
            fi_data=R['dfi'][sd].sort_values('importance',ascending=True)
        else:
            st.subheader("🧠 Feature Importance — 전체")
            fi_data=R['fi'].sort_values('importance',ascending=True)
            if sd!='전체' and sd not in R['dfi']:
                st.caption(f"⚠️ {sd} 조직은 퇴직자 수가 적어 개별 모델 학습이 어렵습니다. 전체 결과를 표시합니다.")
        fig,ax=plt.subplots(figsize=(14,7))
        ax.barh(fi_data['feature'],fi_data['importance'],color=plt.cm.viridis(np.linspace(0.3,0.9,len(fi_data))),height=0.5)
        for i,(f,v) in enumerate(zip(fi_data['feature'],fi_data['importance'])):ax.text(v+0.002,i,f'{v:.3f}',va='center',fontsize=11,fontweight='bold')
        ax.set_title('Feature Importance (Random Forest)',fontsize=16,fontweight='bold');ax.set_xlabel('Importance',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        top3=fi_data.nlargest(3,'importance')
        insight(f"**이탈 예측 핵심 변수 Top 3:** {', '.join([f'{r.feature}({r.importance:.1%})' for _,r in top3.iterrows()])} — 이 변수들이 퇴직 여부를 예측할 때 가장 큰 영향력을 가집니다. 단, 이 수치는 '얼마나 중요한가'만 보여주며, '어떤 방향으로 영향을 미치는가'는 아래 SHAP에서 확인할 수 있습니다.")

        st.markdown("---")

        # 서베이 비교
        if sd!='전체' and sd in R['dsv']:
            st.subheader(f"🎯 재직자 vs 퇴직자 서베이 — {sd}")
            sc=R['dsv'][sd];sc.index=['Active','Terminated']
        else:
            st.subheader("🎯 재직자 vs 퇴직자 서베이 — 전체")
            sc=R['scomp'];sc.index=['Active','Terminated']
        fig,ax=plt.subplots(figsize=(14,6));x=np.arange(3);w=0.3
        bars1=ax.bar(x-w/2,sc.iloc[0],w,label='Active (재직자)',color='#3b82f6')
        bars2=ax.bar(x+w/2,sc.iloc[1],w,label='Terminated (퇴직자)',color='#ef4444')
        ax.set_xticks(x);ax.set_xticklabels(['Engagement\nScore','Satisfaction\nScore','Work-Life\nBalance Score'],fontsize=12)
        ax.set_ylabel('Average Score',fontsize=12);ax.legend(fontsize=12);ax.set_ylim(2.0,4.0)
        for b in bars1:ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.03,f'{b.get_height():.2f}',ha='center',fontsize=11,fontweight='bold',color='#3b82f6')
        for b in bars2:ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.03,f'{b.get_height():.2f}',ha='center',fontsize=11,fontweight='bold',color='#ef4444')
        ax.set_title('Active vs Terminated Survey Comparison',fontsize=16,fontweight='bold');ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        diff=sc.iloc[1]-sc.iloc[0]
        cols=['Engagement Score','Satisfaction Score','Work-Life Balance Score']
        neg=[c.replace(' Score','') for c,d in zip(cols,diff) if d<0]
        if neg:
            warn_insight(f"**퇴직자 서베이 경고:** 퇴직자는 재직자 대비 {', '.join(neg)} 점수가 낮습니다. 이 항목들의 하락이 이탈 선행 지표로 작동하므로, 정기 서베이에서 해당 점수가 떨어지는 직원/조직을 조기에 감지하는 체계가 필요합니다.")

        st.markdown("---")

        # SHAP
        if sd!='전체':
            st.subheader(f"📊 SHAP Summary Plot — {sd}")
            dm=R['odepts']==sd
            if dm.sum()>=10:
                plt.figure(figsize=(14,7));shap.summary_plot(R['osc1'][dm],R['Xosd'][dm],plot_type='dot',show=False)
                st.pyplot(plt.gcf());plt.close('all')
            else:
                st.caption(f"⚠️ {sd} 샘플 10건 미만, 전체 SHAP 표시")
                plt.figure(figsize=(14,7));shap.summary_plot(R['osc1'],R['Xosd'],plot_type='dot',show=False)
                st.pyplot(plt.gcf());plt.close('all')
        else:
            st.subheader("📊 SHAP Summary Plot — 전체")
            plt.figure(figsize=(14,7));shap.summary_plot(R['osc1'],R['Xosd'],plot_type='dot',show=False)
            st.pyplot(plt.gcf());plt.close('all')
        insight("**SHAP 해석:** 각 점은 하나의 직원입니다. 빨간 점(높은 값)이 오른쪽에 몰려있으면 '해당 변수 값이 높을수록 퇴직 확률 증가', 파란 점(낮은 값)이 오른쪽이면 '값이 낮을수록 퇴직 확률 증가'를 의미합니다. Feature Importance가 '무엇이 중요한가'라면, SHAP은 '어떻게 영향을 미치는가'를 보여줍니다.")

        st.markdown("---")

        # 교차 이탈률
        if sd!='전체':
            st.subheader(f"🔍 {sd} 직무별 이탈률")
            csd=R['cs'][(R['cs']['DepartmentType']==sd)&(R['cs']['total']>=5)].sort_values('rate',ascending=False)
        else:
            st.subheader("🔍 조직 × 직무 교차 이탈률 (Top 15)")
            csd=R['cs'][R['cs']['total']>=10].nlargest(15,'rate')
        st.dataframe(csd[['DepartmentType','Title','total','terminated','rate']].rename(columns={'DepartmentType':'Dept','Title':'Title','total':'Total','terminated':'Term','rate':'Rate(%)'}),use_container_width=True,hide_index=True,height=400)
        insight("**교차 이탈률:** 조직과 직무를 동시에 고려하여 가장 위험한 그룹을 식별합니다. 인원이 많으면서 이탈률도 높은 그룹이 비즈니스 영향이 가장 크므로 우선 대응 대상입니다.")

    # ═══ TAB 3 ═══
    with tab3:
        st.subheader("📋 Confusion Matrix")
        fig,ax=plt.subplots(figsize=(10,8))
        sns.heatmap(R['cm'],annot=True,fmt='d',cmap='Blues',ax=ax,annot_kws={'size':18},xticklabels=['Active Predicted','Terminated Predicted'],yticklabels=['Actual Active','Actual Terminated'])
        ax.set_title('Confusion Matrix',fontsize=16,fontweight='bold');ax.tick_params(labelsize=12)
        plt.tight_layout();st.pyplot(fig);plt.close()
        tn,fp,fn,tp=R['cm'].ravel()
        insight(f"**해석:** 실제 퇴직자 중 모델이 잡아낸 비율(Recall) = {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%. 실제 퇴직자를 놓치는 건수(False Negative) = {fn}건. HR 관점에서 이 수치가 낮을수록 위험 직원을 놓칠 확률이 줄어듭니다.")

        st.markdown("---")

        st.subheader("📊 성능 지표")
        st.metric("ROC-AUC Score",f"{R['auc']:.4f}")
        st.dataframe(pd.DataFrame(R['rpt']).T.round(3),use_container_width=True)
        insight(f"**ROC-AUC {R['auc']:.4f}:** 0.5가 무작위, 1.0이 완벽 예측입니다. 현재 수준은 HR 이탈 예측에서 우수한 성능이며, 조직 수준의 패턴 분석과 위험 그룹 식별에 신뢰할 수 있는 수준입니다.")

    # ═══ TAB 4 ═══
    with tab4:
        st.subheader("👤 Employee Risk Scoring")
        st.caption("Random Forest 모델이 각 직원의 이탈 확률을 0~100점으로 산출합니다.")
        m=R['merged'].copy()
        if sd!='전체':m=m[m['DepartmentType']==sd]
        if stl!='전체':m=m[m['Title']==stl]
        c1,c2,c3=st.columns(3)
        h=len(m[m['Risk_Score']>=60]);mid=len(m[(m['Risk_Score']>=30)&(m['Risk_Score']<60)]);lo=len(m[m['Risk_Score']<30])
        c1.metric("🔴 High Risk (60+)",f"{h}명");c2.metric("🟡 Medium (30-59)",f"{mid}명");c3.metric("🟢 Low (0-29)",f"{lo}명")

        st.markdown("---")

        st.subheader("📊 위험 점수 분포")
        fig,ax=plt.subplots(figsize=(14,5))
        ax.hist(m['Risk_Score'],bins=20,color='#3b82f6',alpha=0.7,edgecolor='white')
        ax.axvline(60,color='#ef4444',ls='--',lw=2.5,label='High Risk (60)');ax.axvline(30,color='#f59e0b',ls='--',lw=2.5,label='Medium Risk (30)')
        ax.set_title('Risk Score Distribution',fontsize=16,fontweight='bold');ax.set_xlabel('Risk Score',fontsize=12);ax.set_ylabel('Count',fontsize=12);ax.legend(fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        insight(f"**위험 분포:** 고위험(60+) {h}명, 중위험(30-59) {mid}명, 저위험(0-29) {lo}명. 고위험 직원은 즉시 1:1 면담과 리텐션 조치가 필요하며, 중위험 직원은 정기 모니터링 대상입니다.")

        st.markdown("---")

        st.subheader("🏢 조직별 평균 위험 점수")
        dr=m.groupby('DepartmentType')['Risk_Score'].mean().sort_values(ascending=True)
        fig,ax=plt.subplots(figsize=(14,5))
        ax.barh(dr.index,dr.values,color='#8b5cf6',height=0.5)
        for i,v in enumerate(dr.values):ax.text(v+0.5,i,f'{v:.1f}',va='center',fontsize=12,fontweight='bold')
        ax.set_title('Average Risk Score by Department',fontsize=16,fontweight='bold');ax.set_xlabel('Avg Risk Score',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()

        st.markdown("---")

        st.subheader("🔴 고위험 직원 목록 (Risk Score ≥ 60)")
        cols=['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating']
        avail=[c for c in cols if c in m.columns]
        hr=m[m['Risk_Score']>=60].nlargest(50,'Risk_Score')
        if len(hr)>0:st.dataframe(hr[avail],use_container_width=True,hide_index=True)
        else:st.success("✅ No high-risk employees.")

        st.markdown("---")

        st.subheader("📋 전체 직원 위험도 테이블")
        search=st.text_input("🔍 사번, 직무 등으로 검색")
        full=m[avail].sort_values('Risk_Score',ascending=False)
        if search:full=full[full.astype(str).apply(lambda x:x.str.contains(search,case=False)).any(axis=1)]
        st.dataframe(full,use_container_width=True,hide_index=True,height=500)

    # ═══ TAB 5 ═══
    with tab5:
        st.subheader("🎯 HR Action Plan")
        if sd!='전체' and sd in R['dfi']:fis=R['dfi'][sd]
        else:fis=R['fi']
        fi_text=', '.join([f"{r['feature']}({r['importance']:.3f})" for _,r in fis.head(5).iterrows()])
        ctx=f"Total: {len(emp)}, Term: {emp['Attrition'].sum()}, Rate: {emp['Attrition'].mean()*100:.1f}%, Top Features: {fi_text}"
        dl=R['ds'].sort_values('rate',ascending=False)
        if sd!='전체':dl=dl[dl['DepartmentType']==sd]
        for _,r in dl.iterrows():
            rate=r['rate']
            with st.expander(f"📋 {r['DepartmentType']} — {rate}%",expanded=(rate>=10)):
                mc1,mc2,mc3=st.columns(3);mc1.metric("Total",f"{r['total']}");mc2.metric("Term",f"{int(r['terminated'])}");mc3.metric("Rate",f"{rate}%")
                if ak:
                    if st.button(f"🧠 AI Plan - {r['DepartmentType']}",key=f"ai_{r['DepartmentType']}"):
                        with st.spinner("Generating..."):st.markdown(get_ai_plan(ak,r['DepartmentType'],rate,r['total'],fi_text,ctx))
                else:
                    if rate>=15:st.markdown("🔴 **[Urgent]** Retention package & 1:1 interviews\n\n🟡 **[High]** Career path transparency & Work-life balance\n\n🔵 **[Mid]** Culture improvement & communication")
                    elif rate>=5:st.markdown("🟡 **[High]** Target high-turnover roles\n\n🔵 **[Mid]** Mentoring & Regular survey")
                    else:st.markdown("🟢 **[Maintain]** Current policy\n\n🔵 **[Mid]** Benchmark best practices")
                if not ak:st.caption("💡 사이드바에서 OpenAI API Key 입력 시 AI 맞춤 액션 플랜 생성")

    # ═══ TAB 6 ═══
    with tab6:
        st.subheader("📥 보고서 다운로드")
        rd=st.selectbox("대상 조직",['All']+sorted(R['ds']['DepartmentType'].tolist()),key="rpt")
        if st.button("📄 PDF 보고서 생성",type="primary",use_container_width=True):
            with st.spinner("PDF 생성 중..."):
                pdf=gen_pdf(R,rd)
                st.download_button("⬇️ PDF 다운로드",pdf,f"Report_{rd}.pdf","application/pdf",use_container_width=True)
        st.markdown("---")
        st.subheader("📊 CSV 다운로드")
        csv1=R['merged'][['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating','Attrition']].to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ 직원 위험도 CSV",csv1,"risk_scores.csv","text/csv",use_container_width=True)
        csv2=R['ds'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ 조직별 통계 CSV",csv2,"dept_stats.csv","text/csv",use_container_width=True)

if __name__=="__main__":main()
