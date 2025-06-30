# Cyber Risk Bayesian System (Based on Technical Guide)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import streamlit as st

# Generate Simulated Dataset (500 rows)
def generate_dataset(n=500):
    np.random.seed(42)
    from scipy.stats import nbinom, bernoulli, pareto, lognorm, beta, gamma, expon, weibull_min, truncnorm

    phishing_attempts = np.random.poisson(5, size=n)
    phishing_attempts[np.random.rand(n) < 0.3] = 0
    ddos_intensity = weibull_min(c=1.5, scale=2.0).rvs(n)
    malware_infections = nbinom(n=2, p=0.3).rvs(n)
    insider_threats = bernoulli(0.15).rvs(n)
    direct_fraud_loss = pareto(b=1.2, scale=1000).rvs(n)
    downtime_costs = lognorm(s=1.5, scale=np.exp(8.0)).rvs(n)
    customer_churn_rate = beta(2, 8).rvs(n)
    regulatory_fines = np.where(np.random.rand(n) < 0.7,
                                np.random.normal(loc=50000, scale=10000, size=n),
                                pareto(b=1.2, scale=100000).rvs(n))
    mobile_money_volume = gamma(a=3, scale=1.5).rvs(n)
    security_investment = truncnorm((0 - 200000) / 50000, np.inf, loc=200000, scale=50000).rvs(n)
    employee_training = beta(5, 2).rvs(n)
    load_shedding_frequency = expon(scale=1/0.2).rvs(n)

    account_takeover = 1 / (1 + np.exp(-0.5 * phishing_attempts))
    ddos_downtime_costs = ddos_intensity * 1000 + np.random.normal(0, 0.1, n)
    fraud_loss_from_insider = gamma(a=2, scale=1).rvs(n) * insider_threats

    return pd.DataFrame({
        'Phishing_Attempts': phishing_attempts,
        'DDoS_Intensity': ddos_intensity,
        'Malware_Infections': malware_infections,
        'Insider_Threats': insider_threats,
        'Direct_Fraud_Loss': direct_fraud_loss,
        'Downtime_Costs': downtime_costs,
        'Customer_Churn_Rate': customer_churn_rate,
        'Regulatory_Fines': regulatory_fines,
        'MobileMoney_Volume': mobile_money_volume,
        'Security_Investment': security_investment,
        'Employee_Training': employee_training,
        'Load_Shedding_Frequency': load_shedding_frequency,
        'Account_Takeover_Prob': account_takeover,
        'DDoS_Downtime_Costs': ddos_downtime_costs,
        'Fraud_Loss_From_Insider': fraud_loss_from_insider
    })

# Model interpretation in plain language
def generate_interpretation(summary_df):
    interpretation = "### 🔍 Model Interpretation Report\n\n"

    # Extract means
    fraud_loss_mean = summary_df.loc['fraud_loss', 'mean']
    downtime_loss_mean = summary_df.loc['downtime_loss', 'mean']
    total_impact_mean = summary_df.loc['financial_impact', 'mean']

    # Interpret Fraud Loss
    if fraud_loss_mean < 1000:
        interpretation += "🟢 **Fraud Loss** is expected to be low, possibly due to effective phishing controls or low insider threat exposure.\n\n"
    elif fraud_loss_mean < 5000:
        interpretation += "🟠 **Fraud Loss** is moderate. You may want to evaluate fraud detection and staff background checks.\n\n"
    else:
        interpretation += "🔴 **Fraud Loss** is high! Consider strengthening fraud risk controls and phishing prevention.\n\n"

    # Interpret Downtime Loss
    if downtime_loss_mean < 500:
        interpretation += "🟢 **Downtime Costs** are minimal. Good resilience or DDoS protection likely in place.\n\n"
    elif downtime_loss_mean < 2000:
        interpretation += "🟠 **Downtime Costs** are moderate. Consider reviewing your incident response plan.\n\n"
    else:
        interpretation += "🔴 **Downtime Costs** are high. Immediate investment in DDoS mitigation may be warranted.\n\n"

    # Interpret Overall Financial Impact
    interpretation += f"💰 **Estimated Total Financial Impact**: **${total_impact_mean:,.2f}**.\n\n"
    interpretation += "_This result is generated using a Bayesian simulation of cyber risks based on your data._"

    return interpretation

# Bayesian Inference Model
@st.cache_data
def run_model(data):
    with pm.Model() as model:
        phishing = pm.Normal('phishing', mu=data['Phishing_Attempts'].mean(), sigma=data['Phishing_Attempts'].std())
        insider = pm.Bernoulli('insider', p=data['Insider_Threats'].mean())
        ddos = pm.Normal('ddos', mu=data['DDoS_Intensity'].mean(), sigma=data['DDoS_Intensity'].std())

        # Outputs
        fraud_loss = pm.Deterministic('fraud_loss', phishing * 100 + insider * 1000)
        downtime_loss = pm.Deterministic('downtime_loss', ddos * 800)

        financial_impact = pm.Normal('financial_impact', mu=fraud_loss + downtime_loss, sigma=500)
        trace = pm.sample(1000, tune=500, return_inferencedata=True, target_accept=0.9)
    return trace

# Streamlit Dashboard
st.set_page_config(page_title="Cyber Risk Bayesian Simulator", page_icon="🔐")
st.title("Cyber Risk Simulation and Quantification")

if st.button("Generate and Analyze Dataset"):
    df = generate_dataset()
    st.success("✅ Simulated Data Generated Successfully")
    st.dataframe(df.head())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Running Bayesian Inference Model")
    trace = run_model(df)

    st.write("📋 Posterior Summary")
    summary = az.summary(trace)
    st.dataframe(summary)

    st.subheader("📌 Interpretation")
    report = generate_interpretation(summary)
    st.markdown(report)

    st.subheader("📈 Distributions")
    az.plot_posterior(trace, var_names=["fraud_loss", "downtime_loss", "financial_impact"])
    st.pyplot(plt.gcf())

    st.subheader("🔗 Pair Plot")
    az.plot_pair(trace, var_names=["fraud_loss", "downtime_loss"], kind="kde")
    st.pyplot(plt.gcf())
