# SentinelX AI — Cybersecurity Intelligence & Threat Detection Platform

SentinelX AI is a portfolio-grade cybersecurity analytics dashboard demonstrating machine-learning threat classification, neural-network modelling, anomaly detection, simulated event monitoring, IOC triage and network telemetry exploration.

## Key capabilities
- **Threat Detection:** score network-session records with Random Forest; inspect risk probability, risk band and anomaly flags.
- **Neural-Network Model:** compare a multilayer perceptron (MLP) classifier against Random Forest.
- **Anomaly Analytics:** Isolation Forest flags unusual observations for analyst review.
- **Live Monitoring Simulation:** generate synthetic event batches, populate a session-only queue and export results.
- **Threat Intelligence Desk:** maintain a manual indicator-of-compromise register with severity and triage status.
- **Data Explorer:** inspect, visualize and download generated data or upload compatible CSV telemetry.
- **Model Lab:** view held-out metrics, confusion matrices and feature importance.

## Scope and limitations
This is an educational demonstration, not a production security product. Data and event streams are synthetic. The IOC register is manual and does not connect to an external threat feed. The app does not monitor real networks, block traffic, remediate incidents or take external actions. Model scores use generated labels and are not real-world performance claims. Do not upload confidential production telemetry to a public deployment.

## Run locally
```bash
git clone https://github.com/YOUR_USERNAME/SentinelX-AI.git
cd SentinelX-AI
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Render
1. Push this repository to GitHub.
2. In Render, create a Blueprint and select this repository, or create a Web Service.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. After deployment, set `SENTINELX_PUBLIC_URL` in Render environment variables to your public `https://...onrender.com` URL. The footer will then show **Open Live Dashboard**.

## Project structure
```text
SentinelX-AI/
├── app.py
├── requirements.txt
├── render.yaml
├── README.md
├── LICENSE
├── .gitignore
├── assets/
│   └── sentinelx_logo.png
└── .streamlit/config.toml
```

## Technology stack
Python, Streamlit, Pandas, NumPy, Scikit-learn and Plotly.

## LinkedIn project entry
**Title:** SentinelX AI — Cybersecurity Intelligence & Threat Detection Platform

**Description:** Developed an interactive cybersecurity analytics platform showcasing machine-learning threat classification, neural-network modelling, anomaly detection, simulated event monitoring, IOC triage and network telemetry visualization. Built with Python, Streamlit and Scikit-learn.

**Suggested skills:** Cybersecurity Analytics · Machine Learning · Anomaly Detection · Python · Streamlit

## GitHub
Suggested repository name: `SentinelX-AI`

Suggested topics: `cybersecurity`, `machine-learning`, `threat-detection`, `anomaly-detection`, `network-security`, `python`, `streamlit`, `scikit-learn`, `security-analytics`, `data-visualization`.

## Author
**Mohamed Riyaz S.**  
Integrated M.Tech CSE | Business Analytics  
VIT Chennai

## License
MIT
