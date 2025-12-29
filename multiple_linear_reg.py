import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# ---------------- Page config ----------------
st.set_page_config(
    page_title="Multiple Linear Regression",
    layout="centered"
)
# ---------------- Load CSS ----------------
def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
# ---------------- Title ----------------
st.markdown(
    """
    <div class="card">
        <h1>Multiple Linear Regression App</h1>
        <p><b>Predict Tip Amount using Total Bill and Table Size</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
# ---------------- Load dataset ----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
data = load_data()
# ---------------- Dataset preview ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(data[['total_bill', 'size', 'tip']].head())
st.markdown('</div>', unsafe_allow_html=True)
# ---------------- Prepare data ----------------
X = data[['total_bill', 'size']]
y = data['tip']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ---------------- Train model ----------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
# ---------------- Evaluation ----------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
# ---------------- Metrics display ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Evaluation(Performance) Metrics")
st.markdown(
    f"""
    - **R² Score:** {r2:.4f}  
    - **Adjusted R²:** {adjusted_r2:.4f}  
    - **MAE:** {mae:.4f}  
    - **MSE:** {mse:.4f}  
    - **RMSE:** {rmse:.4f}
    """
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Visualization ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip (Size Fixed) (with multiple regression)")

fig, ax = plt.subplots()

# scatter actual data
ax.scatter(
    data['total_bill'],
    data['tip'],
    alpha=0.5,
    label="Actual Data",
    color="blue"
)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
ax.plot(data['total_bill'],model.predict(scaler.transform(data[['total_bill','size']])),color='red',label='Regression Line')

# fix size to mean for plotting
avg_size = data['size'].mean()

X_plot = pd.DataFrame({
    "total_bill": data['total_bill'],
    "size": avg_size
})
X_plot_scaled = scaler.transform(X_plot)
y_plot = model.predict(X_plot_scaled)
ax.legend()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Coefficients ----------------
st.markdown(
    f"""
    <div class="card">
        <h3>Model Parameters</h3>
        <p><b>Intercept:</b> {model.intercept_:.4f}</p>
        <p><b>Coefficient (Total Bill):</b> {model.coef_[0]:.4f}</p>
        <p><b>Coefficient (Size):</b> {model.coef_[1]:.4f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Prediction Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")
total_bill = st.number_input(
    "Enter Total Bill Amount:",
    min_value=float(data['total_bill'].min()),
    max_value=float(data['total_bill'].max()),
    value=30.0,
    step=1.0
)
size = st.number_input(
    "Enter Table Size:",
    min_value=int(data['size'].min()),
    max_value=int(data['size'].max()),
    value=2,
    step=1
)
input_df = pd.DataFrame(
    [[total_bill, size]],
    columns=["total_bill", "size"]
)
input_scaled = scaler.transform(input_df)
predicted_tip = model.predict(input_scaled)[0]

st.markdown(f"""
    <div class="prediction-box">
        Predicted Tip Amount
        <br>
        <b>${predicted_tip:.2f}</b>
    </div>
    """,unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
