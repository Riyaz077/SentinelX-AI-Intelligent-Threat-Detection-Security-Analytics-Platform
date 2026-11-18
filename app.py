import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title='SentinelX AI', page_icon='assets/sentinelx_logo.png', layout='wide')
FEATURES=['session_duration','packet_size','failed_logins','connection_rate','port_risk','geo_risk_score','night_activity']

st.markdown('''
<style>
:root {
    --sx-bg: #101411;
    --sx-panel: #19251E;
    --sx-green: #39D98A;
    --sx-gold: #D6BD8A;
    --sx-text: #F2F0E8;
    --sx-muted: #A4B2A5;
}

.stApp {
    background: linear-gradient(145deg, #101411, #151E18, #101411);
    color: #F2F0E8;
}

[data-testid="stSidebar"] {
    background: #121A15;
    border-right: 1px solid rgba(57,217,138,0.14);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #F2F0E8 !important;
}

p, label, li {
    color: #C4CEC4;
}

.sx-hero {
    padding: 28px 30px;
    background: linear-gradient(120deg, #1B2B21, #111A14);
    border: 1px solid rgba(57,217,138,0.20);
    border-radius: 18px;
    margin-bottom: 20px;
}

.sx-eyebrow {
    color: #39D98A;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.sx-title {
    color: #F2F0E8;
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0;
}

.sx-sub {
    color: #A4B2A5;
    font-size: 15px;
}

.sx-kpi {
    background: linear-gradient(145deg, #1B2B21, #17231B);
    border: 1px solid rgba(214,189,138,0.14);
    border-radius: 14px;
    padding: 17px 18px;
    min-height: 105px;
}

.sx-label {
    color: #A4B2A5;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sx-value {
    color: #F2F0E8;
    font-size: 27px;
    font-weight: 750;
    margin-top: 7px;
}

.sx-note {
    color: #39D98A;
    font-size: 11px;
}

.sx-footer {
    text-align: center;
    padding: 34px 12px 12px;
    margin-top: 38px;
    border-top: 1px solid rgba(214,189,138,0.18);
}

.sx-footer-title {
    color: #39D98A;
    font-size: 16px;
    font-weight: 750;
    margin-bottom: 20px;
}

.sx-footer-label {
    color: #A4B2A5;
    font-size: 11px;
    margin-bottom: 12px;
}

.sx-footer-name {
    color: #F2F0E8;
    font-size: 25px;
    font-weight: 750;
    line-height: 1.25;
    margin-bottom: 14px;
}

.sx-footer-meta {
    color: #A4B2A5;
    font-size: 14px;
    line-height: 1.5;
    margin: 0 0 8px;
}

.sx-footer-copy {
    color: #8B9A8D;
    font-size: 10px;
    padding-top: 20px;
    margin-top: 22px;
    border-top: 1px solid rgba(214,189,138,0.18);
}

/* Centered workspace navigation */
div[data-testid="stSegmentedControl"] {
    display: flex;
    justify-content: center;
    width: 100%;
    margin: 4px auto 24px;
}

div[data-testid="stSegmentedControl"] [role="radiogroup"] {
    justify-content: center;
    gap: 7px;
    background: #19251E;
    border: 1px solid rgba(214,189,138,0.18);
    padding: 7px;
    border-radius: 13px;
    flex-wrap: wrap;
}

div[data-testid="stSegmentedControl"] button {
    border-radius: 9px !important;
    padding: 10px 18px !important;
    min-height: 44px;
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: #39D98A !important;
    color: #101411 !important;
    border: 1px solid #39D98A !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #39D98A;
    color: #101411;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #D6BD8A;
    color: #101411;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #19251E;
    border: 1px solid rgba(57,217,138,0.14);
    border-radius: 12px;
    padding: 14px;
}
</style>
''', unsafe_allow_html=True)


def make_data(n=2400,seed=42):
 r=np.random.default_rng(seed); d=r.gamma(2,2.2,n); p=r.normal(620,210,n).clip(40,2200); f=r.poisson(.35,n); c=r.gamma(2,5,n); pr=r.choice([0,1,2,3],n,p=[.62,.22,.12,.04]); g=r.beta(1.2,5,n); night=r.binomial(1,.22,n)
 score=.13*d+.0012*p+.65*f+.12*c+.7*pr+2.2*g+.9*night; y=(score>np.quantile(score,.84)).astype(int)
 return pd.DataFrame(dict(session_duration=d.round(2),packet_size=p.round(1),failed_logins=f,connection_rate=c.round(2),port_risk=pr,geo_risk_score=g.round(3),night_activity=night,threat_label=y))

@st.cache_resource
def train():
 df=make_data(); X=df[FEATURES]; y=df.threat_label; xa,xb,ya,yb=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
 rf=RandomForestClassifier(n_estimators=180,max_depth=10,class_weight='balanced',random_state=42).fit(xa,ya)
 mlp=Pipeline([('scale',StandardScaler()),('mlp',MLPClassifier(hidden_layer_sizes=(64,32),max_iter=350,early_stopping=True,random_state=42))]).fit(xa,ya)
 iso=IsolationForest(contamination=.12,n_estimators=160,random_state=42).fit(xa)
 return df,rf,mlp,iso,xb,yb

df,rf,mlp,iso,Xtest,ytest=train()
def score(frame):
 out=frame.copy(); out['risk_probability']=rf.predict_proba(frame[FEATURES])[:,1]; out['risk_level']=np.where(out.risk_probability>=.75,'Critical',np.where(out.risk_probability>=.5,'High',np.where(out.risk_probability>=.25,'Moderate','Low'))); out['anomaly_flag']=(iso.predict(frame[FEATURES])==-1).astype(int); return out
def kpi(label,value,note): st.markdown(f'<div class="sx-kpi"><div class="sx-label">{label}</div><div class="sx-value">{value}</div><div class="sx-note">{note}</div></div>',unsafe_allow_html=True)

with st.sidebar:
 st.image('assets/sentinelx_logo.png',width=112)
 st.markdown('## SentinelX AI'); st.caption('Cybersecurity Intelligence'); st.divider()
 st.markdown('**SYSTEM STATUS**')
 st.success('Detection engine initialized')
 st.caption('Synthetic demonstration environment')
 st.caption('No external systems are accessed.')

st.markdown('<div class="sx-hero"><div class="sx-eyebrow">CYBERSECURITY INTELLIGENCE PLATFORM</div><div class="sx-title">SentinelX AI</div><div class="sx-sub">Machine-learning-powered threat detection, anomaly analytics and security intelligence in one interactive workspace.</div></div>',unsafe_allow_html=True)

nav_left, nav_center, nav_right = st.columns([0.25, 9.5, 0.25])
with nav_center:
 page = st.segmented_control(
  'Workspace',
  options=['Overview','Threat Detection','Live Monitoring','Threat Intelligence','Model Lab','Data Explorer','About & Governance'],
  default='Overview',
  label_visibility='collapsed',
  key='sentinelx_workspace',
  width='stretch'
 )

if 'events' not in st.session_state: st.session_state.events=pd.DataFrame()
if 'iocs' not in st.session_state: st.session_state.iocs=pd.DataFrame([{'Indicator':'198.51.100.24','Type':'IPv4','Context':'Documentation-range sample IOC','Severity':'High','Status':'Investigating'},{'Indicator':'suspicious-example.test','Type':'Domain','Context':'Reserved example domain','Severity':'Medium','Status':'New'}])

if page=='Overview':
 s=score(df); c=st.columns(4)
 with c[0]: kpi('Events Analysed',f'{len(s):,}','Synthetic network sessions')
 with c[1]: kpi('Threat Events',f'{int(s.threat_label.sum()):,}',f'{s.threat_label.mean():.1%} of sample')
 with c[2]: kpi('High / Critical',f'{int((s.risk_probability>=.5).sum()):,}','Model-assessed risk')
 with c[3]: kpi('Anomalies',f'{int(s.anomaly_flag.sum()):,}','Isolation Forest flags')
 a,b=st.columns(2)
 with a: st.markdown('#### Threat risk distribution'); st.plotly_chart(px.histogram(s,x='risk_level',color='risk_level',template='plotly_dark'),use_container_width=True)
 with b: st.markdown('#### Risk probability profile'); st.plotly_chart(px.histogram(s,x='risk_probability',nbins=28,color_discrete_sequence=['#20D7F4'],template='plotly_dark'),use_container_width=True)
 st.markdown('#### Highest-risk sessions'); st.dataframe(s.sort_values('risk_probability',ascending=False).head(10).round(3),use_container_width=True,hide_index=True)
 st.info('Generated demonstration data only. Metrics are illustrative and do not demonstrate protection against live attacks.')
elif page=='Threat Detection':
 st.subheader('Threat Detection Workspace'); st.write('Score network-session records using the trained Random Forest and Isolation Forest models.')
 st.caption('Required CSV fields: '+', '.join(FEATURES)); up=st.file_uploader('Upload network-session CSV',type=['csv'])
 if up:
  try:
   raw=pd.read_csv(up); missing=[x for x in FEATURES if x not in raw]
   if missing: st.error('Missing columns: '+', '.join(missing))
   else:
    result=score(raw); st.dataframe(result.sort_values('risk_probability',ascending=False),use_container_width=True,hide_index=True); st.download_button('Download scored events',result.to_csv(index=False).encode(),'sentinelx_scored_events.csv')
  except Exception as e: st.error(f'Could not read CSV: {e}')
 st.markdown('#### Manual session assessment'); defaults=[4.2,780,2,12,2,.65,1]
 with st.form('manual'):
  cols=st.columns(3); vals={}
  for i,f in enumerate(FEATURES):
   with cols[i%3]: vals[f]=st.number_input(f.replace('_',' ').title(),value=float(defaults[i]))
  go=st.form_submit_button('Assess session',type='primary')
 if go:
  r=score(pd.DataFrame([vals])).iloc[0]; st.metric('Estimated threat probability',f'{r.risk_probability:.1%}'); st.warning(f"Risk classification: {r.risk_level}"); st.write('Anomaly indicator:', 'Flagged' if r.anomaly_flag else 'Not flagged'); st.caption('Decision-support signal only, not a definitive security verdict.')
elif page=='Live Monitoring':
 st.subheader('Live Event Monitoring'); st.write('Replay generated events to demonstrate continuous scoring. This is a local simulation, not a live network sensor.')
 n=st.select_slider('Events per batch',[5,10,20,40],value=10)
 if st.button('Generate event batch',type='primary'):
  batch=score(make_data(n,seed=int(datetime.now().timestamp())%100000)); batch.insert(0,'event_time',datetime.now().strftime('%H:%M:%S')); st.session_state.events=pd.concat([batch,st.session_state.events],ignore_index=True).head(500)
 ev=st.session_state.events
 if len(ev):
  c=st.columns(3)
  with c[0]: kpi('Events in queue',len(ev),'Session-only')
  with c[1]: kpi('High / critical',int((ev.risk_probability>=.5).sum()),'Requires review')
  with c[2]: kpi('Anomalies',int(ev.anomaly_flag.sum()),'Unusual behaviour')
  st.dataframe(ev.sort_values('risk_probability',ascending=False),use_container_width=True,hide_index=True); st.download_button('Export event queue',ev.to_csv(index=False).encode(),'sentinelx_event_queue.csv')
  if st.button('Clear event queue'): st.session_state.events=pd.DataFrame(); st.rerun()
 else: st.info('Generate an event batch to populate the monitoring queue.')
elif page=='Threat Intelligence':
 st.subheader('Threat Intelligence Desk'); st.write('Manual indicator-of-compromise register for analyst triage. No external threat feed is queried.')
 with st.form('ioc'):
  a,b,c=st.columns(3); indicator=a.text_input('Indicator'); typ=b.selectbox('Type',['IPv4','Domain','URL','File hash']); sev=c.selectbox('Severity',['Critical','High','Medium','Low']); context=st.text_input('Analyst context'); add=st.form_submit_button('Add indicator')
 if add and indicator.strip():
  st.session_state.iocs.loc[len(st.session_state.iocs)]={'Indicator':indicator,'Type':typ,'Context':context,'Severity':sev,'Status':'New'}; st.success('Indicator added to this session.')
 st.dataframe(st.session_state.iocs,use_container_width=True,hide_index=True); st.download_button('Export IOC register',st.session_state.iocs.to_csv(index=False).encode(),'sentinelx_ioc_register.csv'); st.caption('Sample indicators are illustrative. Validate indicators with trusted sources before operational use.')
elif page=='Model Lab':
 st.subheader('Model Performance & Explainability'); st.write('Compare Random Forest with a scaled multilayer perceptron and inspect feature importance.')
 for name,model in [('Random Forest',rf),('Neural Network (MLP)',mlp)]:
  pred=model.predict(Xtest); st.markdown('#### '+name); c=st.columns(4)
  for col,label,val in zip(c,['Accuracy','Precision','Recall','F1 score'],[accuracy_score(ytest,pred),precision_score(ytest,pred,zero_division=0),recall_score(ytest,pred,zero_division=0),f1_score(ytest,pred,zero_division=0)]):
   with col: st.metric(label,f'{val:.3f}')
  st.plotly_chart(px.imshow(confusion_matrix(ytest,pred),text_auto=True,x=['Predicted benign','Predicted threat'],y=['Actual benign','Actual threat'],template='plotly_dark',title='Confusion matrix — '+name),use_container_width=True)
 imp=pd.DataFrame({'Feature':FEATURES,'Importance':rf.feature_importances_}).sort_values('Importance'); st.plotly_chart(px.bar(imp,x='Importance',y='Feature',orientation='h',template='plotly_dark',title='Random Forest feature importance'),use_container_width=True)
 st.caption('Scores use a held-out split of generated synthetic data; they are not production security benchmarks.')
elif page=='Data Explorer':
 st.subheader('Network Telemetry Explorer'); fields=st.multiselect('Select fields',list(df.columns),default=list(df.columns)); st.dataframe(df[fields].head(1000),use_container_width=True,hide_index=True)
 a,b=st.columns(2)
 with a:
  field=st.selectbox('Distribution field',FEATURES); st.plotly_chart(px.histogram(df,x=field,color='threat_label',barmode='overlay',template='plotly_dark'),use_container_width=True)
 with b:
  x=st.selectbox('X axis',FEATURES); y=st.selectbox('Y axis',FEATURES,index=1); st.plotly_chart(px.scatter(df.sample(min(700,len(df)),random_state=4),x=x,y=y,color='threat_label',template='plotly_dark'),use_container_width=True)
 st.download_button('Download demonstration dataset',df.to_csv(index=False).encode(),'sentinelx_demo_network_data.csv')
else:
 st.subheader('Platform Scope & Responsible Use')
 st.markdown('''**SentinelX AI** is an educational and portfolio-grade cybersecurity analytics demonstration.

- Supervised threat classification using Random Forest.
- Multilayer perceptron neural-network classifier.
- Unsupervised anomaly detection using Isolation Forest.
- Simulated event replay, risk scoring, alert review and manual IOC register.
- CSV telemetry exploration and downloadable analysis outputs.

**Implementation boundaries:** all built-in data and event streams are synthetic; the IOC register is manual and does not connect to an external feed; no real network monitoring, automated blocking, remediation or external security actions are performed. Models are trained on generated labels and are not validated for real-world attack detection. Avoid uploading confidential production telemetry to a public deployment.

**Technology:** Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly''')

public_url=os.getenv('SENTINELX_PUBLIC_URL','').strip(); link=f'<div class="sx-footer-meta"><a href="{public_url}" target="_blank" style="color:#20D7F4;text-decoration:none;font-size:13px;font-weight:600">Open Live Dashboard ↗</a></div>' if public_url else ''
st.markdown(f'''<div class="sx-footer"><div class="sx-footer-title">SentinelX AI – Cybersecurity Intelligence &amp; Threat Detection Platform</div><div class="sx-footer-label">Designed &amp; Developed by</div><div class="sx-footer-name">Mohamed Riyaz S.</div><div class="sx-footer-meta">Integrated M.Tech CSE | Business Analytics</div><div class="sx-footer-meta">VIT Chennai</div>{link}<div class="sx-footer-copy">© 2026 Mohamed Riyaz S. | All Rights Reserved</div></div>''',unsafe_allow_html=True)
