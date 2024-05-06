import streamlit as st
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os


def run_bayesian_model(df: pd.DataFrame, target: str):
    with pm.Model() as model:
        
        mutable_data = {col: pm.MutableData(f"data_{col}", df[col]) for col in df.columns if col != target}
        target_data = pm.MutableData(f"data_{target}", df[target])
        intercept = pm.Normal('Intercept', mu=0, sigma=10)
        coefficients = {col: pm.Normal(col, mu=0, sigma=10) for col in mutable_data}
        
        mu = intercept + sum(coefficients[col] * mutable_data[col] for col in mutable_data)
        
        sigma = pm.HalfNormal('sigma', 10)
        likelihood = pm.Normal(target, mu=mu, sigma=sigma, observed=target_data)
        
        try:
            trace = pm.sample(500, tune=500, return_inferencedata=True, progressbar=True)
        except Exception as e:
            st.error(f"Error during model sampling: {e}")
            return None
        
        return trace


def generate_pdf_report(trace_summary, posterior_plots, pair_plots):
    report_path = "bayesian_analysis_report.pdf"
    c = canvas.Canvas(report_path, pagesize=letter)
    c.drawString(72, 800, "Bayesian Analysis Report")
    
    c.drawString(72, 780, "Trace Summary")
    y_pos = 760
    for line in trace_summary.to_string().split('\n'):
        c.drawString(72, y_pos, line)
        y_pos -= 20
    
    c.drawString(72, y_pos, "Posterior Plots")
    y_pos -= 20
    for plot in posterior_plots:
        c.drawImage(plot, 72, y_pos, width=400, height=300)
        y_pos -= 320
    
    if pair_plots:
        c.drawString(72, y_pos, "Pair Plots")
        y_pos -= 20
        for plot in pair_plots:
            c.drawImage(plot, 72, y_pos, width=400, height=300)
            y_pos -= 320
    
    c.save()
    return os.path.abspath(report_path)

    
st.title("Bayesian Model Data Visualization")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    st.write("### Uploaded Data")
    df = pd.read_csv(uploaded_file)
    st.write(df)
    
    target = st.selectbox("Select Target Variable", df.columns)
    independent_vars = st.multiselect(
        "Select Independent Variables",
        [col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )
    
    if target and independent_vars:
        try:
            df_clean = df[[target] + independent_vars].drop_duplicates().dropna()
            st.write("### Cleaned Data for Analysis")
            st.write(df_clean)
            
            st.write("### Running Bayesian Model...")
            trace = run_bayesian_model(df_clean, target)
            
            if trace is not None:
                st.write("### Trace Summary")
                trace_summary = az.summary(trace)
                st.write(trace_summary)
                
                st.write("### Posterior Plots")
                posterior_plots = []
                az.plot_trace(trace)
                posterior_plot_path = "posterior_trace_plot.png"
                plt.savefig(posterior_plot_path)
                posterior_plots.append(posterior_plot_path)
                st.pyplot(plt)

                st.write("### Pair Plots")
                all_var_names = list(trace.posterior.data_vars)
                st.write(f"### Available Variables in Trace: {all_var_names}")
                relevant_vars = [f"data_{col}" for col in independent_vars] + [f"data_{target}"]
                
                available_vars = [var for var in relevant_vars if var in all_var_names]
                
                if available_vars:
                    az.plot_pair(trace, var_names=available_vars, kind="kde")
                    pair_plot_path = "pair_plot.png"
                    plt.savefig(pair_plot_path)
                    pair_plots = [pair_plot_path]
                    st.pyplot(plt)
                else:
                    st.warning("No matching variables found for pair plot.")
                    pair_plots = []
                
                # Generate PDF report
                pdf_report_path = generate_pdf_report(trace_summary, posterior_plots, pair_plots)
                st.write("### PDF Report Generated")
                st.markdown(f"[Download PDF Report]({pdf_report_path})")
                
        except ValueError as e:
            st.error(f"ValueError: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
