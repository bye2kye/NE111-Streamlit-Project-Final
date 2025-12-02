import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

title = None
xaxis = None
yaxis = None
data = None

DISTRIBUTIONS = {
    "Normal (norm)": stats.norm,
    "Gamma": stats.gamma,
    "Weibull (weibull_min)": stats.weibull_min,
    "Lognormal": stats.lognorm,
    "Exponential": stats.expon,
    "Beta": stats.beta,
    "Chi-square": stats.chi2,
    "Uniform": stats.uniform,
    "Pareto": stats.pareto,
    "Triangular": stats.triang,
}

# functions ///////////////////////////////     

def parse_input(boxdata):       # converting the text input values to a list of floats
    boxdata = boxdata.replace(",", " ").split()
    i = len(boxdata) 
    x = 0
    while x < i:
        boxdata[x] = float(boxdata[x])
        x += 1
    return boxdata

def csvinput(uploaded_file):
    df = pd.read_csv(uploaded_file)
    data = df.values.flatten().astype(float).tolist()
    return data

def histo(data, title, xaxis, yaxis):        # histogram function
    fig, ax = plt.subplots(facecolor="#0d1118")
    ax.set_facecolor("#0d1118")
    counts, bin_edges, _ = ax.hist(
        data,
        bins="fd",
        color="white",
        edgecolor="lightgrey"
        density = True
    )
    ax.set_title(title,
                 family="DejaVu Sans",
                 color="white",
                 fontweight="bold")
    ax.set_xlabel(xaxis,
                  family="DejaVu Sans",
                  color="white",
                  fontweight="bold")
    ax.set_ylabel(yaxis,
                  family="DejaVu Sans",
                  color="white",
                  fontweight="bold")
    ax.tick_params(axis="both",
                   colors="white",
                   labelcolor="white",
                   which="both",)
    
    for spine in ax.spines.values():  # make little tick thingys white
        spine.set_color("white")
    plt.tight_layout()

    return fig, ax, counts, bin_edges

# display ///////////////////////////////     

st.title("Histogram Tool")      # title

boxdata = st.text_input("Data Input:")            # input textbox for data
uploaded_file = st.file_uploader("Or upload a .csv file")   # dropbox

title = st.text_input("Title:")                  # title
xaxis = st.text_input("Label x-axis:")           # x-axis
yaxis = st.text_input("Label y-axis:")           # y-axis

dist_name = st.selectbox("Select distribution to fit:", list(DISTRIBUTIONS.keys()))
fit_button = st.button("Fit distribution")

# session state ///////////////////////////////

if "last_params" not in st.session_state:
    st.session_state["last_params"] = None
if "last_dist" not in st.session_state:
    st.session_state["last_dist"] = None

# backend stuff ///////////////////////////////    

data = None

# 1) Try text input
if boxdata.strip() != "":
    try:
        data = parse_input(boxdata)
    except ValueError:
        st.error("Please only add numbers (comma & space separated)")
        data = None

#override text input if file uploaded
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        try:
            data = csvinput(uploaded_file)
        except Exception:
            st.error("Error reading CSV file. Please ensure it is properly formatted.")
            data = None
    else:
        st.error("Please upload a valid supported file! (.csv)")
        data = None


#plot data if available
if data is not None and len(data) > 0:
    dist = DISTRIBUTIONS[dist_name]

    fig, ax, counts, bin_edges = histo(data, title, xaxis, yaxis)

    if fit_button:
        # automatic fitting
        params = dist.fit(data)

        st.session_state["last_params"] = params
        st.session_state["last_dist"] = dist_name

        x_vals = np.linspace(min(data), max(data), 1000)
        pdf_vals = dist.pdf(x_vals, *params)

        # overlay fitted PDF
        ax.plot(x_vals, pdf_vals, color="red", linewidth=2, label="Fitted PDF")
        ax.legend()

        # compute simple error metrics (hist vs PDF at bin centers)
        centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        pdf_centers = dist.pdf(centers, *params)
        mse = np.mean((counts - pdf_centers) ** 2)
        max_err = np.max(np.abs(counts - pdf_centers))

        st.write("**Fitted parameters:**", params)
        st.write(f"**MSE (hist vs fitted PDF):** {mse:.6f}")
        st.write(f"**Max abs error:** {max_err:.6f}")

    st.pyplot(fig)  # always show the histogram (with or without fit)

else:
    st.info("Enter data in the text box or upload a .csv file to see the histogram and fit.")


#manual fitting //////////////////////////

if data is not None and st.session_state["last_params"] is not None and st.session_state["last_dist"] == dist_name:
    with st.expander("Manual fitting (adjust parameters)"):
        auto_params = st.session_state["last_params"]
        n_params = len(auto_params)
        data_mean = np.mean(data) if data is not None and len(data) > 0 else 0.0
        data_std = np.std(data) if data is not None and len(data) > 0 else 1.0

        manual_params = []
        for i, p in enumerate(auto_params):
            # Set reasonable bounds for each parameter
            if i == 0:
                # Location/mean parameter
                min_val = data_mean - 3 * data_std
                max_val = data_mean + 3 * data_std
                default_val = float(np.clip(p, min_val, max_val))
            elif i == 1:
                # Scale parameter (must be positive)
                min_val = max(0.01, data_std / 10)
                max_val = max(data_std * 5, p * 2)
                default_val = float(np.clip(abs(p), min_val, max_val))
            else:
                # Shape or other parameters
                min_val = max(-10.0, p - abs(p) - 2.0)
                max_val = min(10.0, p + abs(p) + 2.0)
                default_val = float(np.clip(p, min_val, max_val))

            slider_val = st.slider(
                f"param {i}",
                float(min_val),
                float(max_val),
                float(default_val))
            manual_params.append(slider_val)
        manual_params = tuple(manual_params)

        fig2, ax2, counts2, bin_edges2 = histo(data, title, xaxis, yaxis)
        x_vals2 = np.linspace(min(data), max(data), 500)
        pdf_vals2 = dist.pdf(x_vals2, *manual_params)
        ax2.plot(x_vals2, pdf_vals2, color="cyan", linewidth=2, label="Manual Fit PDF")
        ax2.legend()

        centers2 = 0.5 * (bin_edges2[1:] + bin_edges2[:-1])
        pdf_centers2 = dist.pdf(centers2, *manual_params)
        mse2 = np.mean((counts2 - pdf_centers2) ** 2)
        max_err2 = np.max(np.abs(counts2 - pdf_centers2))

        st.write("**Manual fitted parameters:**", manual_params)
        st.write(f"**MSE (hist vs manual PDF):** {mse2:.6f}")
        st.write(f"**Max abs error:** {max_err2:.6f}")
        st.pyplot(fig2)
