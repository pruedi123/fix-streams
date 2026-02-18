import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Inflation Erosion of Fixed Income", layout="wide")
st.title("Fixed Income Inflation Erosion")


@st.cache_data
def load_monthly_cpi():
    """Load CPI monthly returns."""
    df = pd.read_csv(
        "cpi_data_monthly.csv",
        skiprows=7,
        names=["Date", "CPI_Return"],
        usecols=[0, 1],
    )
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["CPI_Return"] = pd.to_numeric(df["CPI_Return"], errors="coerce")
    df = df.dropna(subset=["CPI_Return"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


monthly_cpi = load_monthly_cpi()
min_date = monthly_cpi["Date"].min()
max_date = monthly_cpi["Date"].max()
min_year = min_date.year
max_year = max_date.year

month_names = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# --- Session state defaults ---
if "solved_target" not in st.session_state:
    st.session_state.solved_target = None

# Apply pending overrides BEFORE widgets render
if "apply_worst" in st.session_state:
    m, y = st.session_state.pop("apply_worst")
    st.session_state.start_month_w = m
    st.session_state.start_year_w = y
if "apply_target" in st.session_state:
    st.session_state.spending_target_w = st.session_state.pop("apply_target")

# --- Sidebar controls ---
st.sidebar.header("Parameters")
annual_income = st.sidebar.number_input(
    "Annual income ($)", min_value=1_000, max_value=10_000_000, value=50_000, step=1_000
)
if "start_month_w" not in st.session_state:
    st.session_state.start_month_w = "January"
start_month_name = st.sidebar.selectbox(
    "Start month", month_names, key="start_month_w",
)
start_month = month_names.index(start_month_name) + 1

if "start_year_w" not in st.session_state:
    st.session_state.start_year_w = 1965
start_year = st.sidebar.number_input(
    "Start year", min_value=min_year, max_value=max_year, key="start_year_w",
)

max_years_available = max_year - start_year
num_years = st.sidebar.number_input(
    "Number of years", min_value=1, max_value=100, value=30,
)
if num_years > max_years_available:
    num_years = max_years_available
    st.sidebar.caption(f"Capped to {max_years_available} years (data limit)")

cola_rate = st.sidebar.number_input(
    "COLA rate (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.25,
    help="Cost-of-Living Adjustment. A fixed annual raise applied to your income each year. "
         "For example, 2% means your nominal income grows 2% per year. Set to 0% for a truly fixed income with no raises.",
) / 100.0

st.sidebar.header("Reserve Strategy")
if "spending_target_w" not in st.session_state:
    st.session_state.spending_target_w = annual_income
real_spending_target = st.sidebar.number_input(
    "Real spending target ($)", min_value=1_000, max_value=10_000_000, step=1_000,
    key="spending_target_w",
    help="How much you want to spend each year in today's dollars, keeping your buying power constant. "
         "If you set this below your income, the surplus goes into a reserve fund to cover future shortfalls when inflation outpaces your income.",
)

reserve_spread = st.sidebar.number_input(
    "Reserve return spread (%)", min_value=0.0, max_value=3.0, value=2.0, step=0.25,
    help="The extra return your reserve fund earns above inflation each year. "
         "For example, 2% means your reserves grow at the inflation rate + 2%. "
         "Think of this as the real return on a conservative investment like bonds or CDs.",
) / 100.0
if st.sidebar.button(
    "Solve Max Sustainable Target",
    help="Finds the highest real spending amount you can maintain for the entire period without running out of reserves. "
         "Solves for both your selected period and the worst historical inflation period of the same length.",
):
    st.session_state.solved_target = "pending"


# --- Calculations ---
def build_periods(cpi_data, start_timestamp, n_years):
    """Build annual inflation periods from a start date for n_years."""
    rows = []
    for yr in range(n_years):
        p_start = start_timestamp + pd.DateOffset(years=yr)
        p_end = start_timestamp + pd.DateOffset(years=yr + 1) - pd.DateOffset(months=1)
        mask = (cpi_data["Date"] >= p_start) & (cpi_data["Date"] <= p_end)
        months = cpi_data.loc[mask]
        if months.empty:
            break
        annual_inf = (1 + months["CPI_Return"]).prod() - 1
        label = f"{p_start.strftime('%b %Y')}–{p_end.strftime('%b %Y')}"
        rows.append({"Period": label, "Period_Start": p_start, "Annual_Inflation": annual_inf})
    return rows


def find_worst_inflation_window(cpi_data, n_years):
    """Scan all possible start months and return the one with the highest inflation CAGR."""
    all_months = sorted(cpi_data["Date"].dt.to_period("M").unique())
    worst_cagr = -999
    worst_start = None
    for pm in all_months:
        candidate_start = pm.to_timestamp()
        rows = build_periods(cpi_data, candidate_start, n_years)
        if len(rows) < n_years:
            continue
        cum = 1.0
        for r in rows:
            cum *= 1 + r["Annual_Inflation"]
        cagr = cum ** (1 / n_years) - 1
        if cagr > worst_cagr:
            worst_cagr = cagr
            worst_start = candidate_start
    return worst_start, worst_cagr


start_dt = pd.Timestamp(year=start_year, month=start_month, day=1)
annual_rows = build_periods(monthly_cpi, start_dt, num_years)

if not annual_rows:
    st.error("No CPI data available for the selected period.")
    st.stop()

period = pd.DataFrame(annual_rows)

nominal = []
real = []
cum_inflation = []
cum_inf = 1.0

for i, row in period.iterrows():
    nom = annual_income * (1 + cola_rate) ** i
    cum_inf *= 1 + row["Annual_Inflation"]
    nominal.append(nom)
    cum_inflation.append(cum_inf)
    real.append(nom / cum_inf)

period["Nominal_Income"] = nominal
period["Cumulative_Inflation"] = cum_inflation
period["Real_Income"] = real

# --- Reserve fund calculations ---
def simulate_reserves(df, target, spread):
    """Simulate reserve strategy. Returns (reserve_balance, actual_real_spending, spending_need, target_met)."""
    reserve_bal = []
    actual_spend = []
    spend_need = []
    met = []
    bal = 0.0
    for i, row in df.iterrows():
        annual_inf = row["Annual_Inflation"]
        nom_income = row["Nominal_Income"]
        cum_inf_val = row["Cumulative_Inflation"]
        if i > 0:
            bal *= 1 + annual_inf + spread
        need = target * cum_inf_val
        spend_need.append(need)
        surplus = nom_income - need
        if surplus >= 0:
            bal += surplus
            actual_spend.append(target)
            met.append(True)
        else:
            shortfall = -surplus
            if bal >= shortfall:
                bal -= shortfall
                actual_spend.append(target)
                met.append(True)
            else:
                bal = 0.0
                actual_spend.append(nom_income / cum_inf_val)
                met.append(False)
        reserve_bal.append(bal)
    return reserve_bal, actual_spend, spend_need, met


def solve_max_target(df, spread):
    """Binary search for the maximum real spending target that sustains reserves through the full period."""
    lo, hi = 0.0, float(df["Nominal_Income"].iloc[0])
    for _ in range(50):
        mid = (lo + hi) / 2
        _, _, _, met = simulate_reserves(df, mid, spread)
        if all(met):
            lo = mid
        else:
            hi = mid
    return math.floor(lo)


if st.session_state.solved_target == "pending":
    # Solve for current period
    current_solved = solve_max_target(period, reserve_spread)
    st.session_state.solved_current = current_solved

    # Find worst inflation period and solve for that too
    worst_start, worst_cagr = find_worst_inflation_window(monthly_cpi, num_years)
    if worst_start is not None:
        worst_rows = build_periods(monthly_cpi, worst_start, num_years)
        worst_df = pd.DataFrame(worst_rows)
        w_nominal, w_cum_inflation = [], []
        w_cum = 1.0
        for i, row in worst_df.iterrows():
            w_nominal.append(annual_income * (1 + cola_rate) ** i)
            w_cum *= 1 + row["Annual_Inflation"]
            w_cum_inflation.append(w_cum)
        worst_df["Nominal_Income"] = w_nominal
        worst_df["Cumulative_Inflation"] = w_cum_inflation
        st.session_state.solved_worst = solve_max_target(worst_df, reserve_spread)
        st.session_state.solved_worst_start = worst_start
        st.session_state.solved_worst_cagr = worst_cagr
    else:
        st.session_state.solved_worst = current_solved
        st.session_state.solved_worst_start = None

    st.session_state.solved_target = "done"

if st.session_state.solved_target == "done":
    current_solved = st.session_state.solved_current
    current_pct = (current_solved / annual_income) * 100
    st.sidebar.success(
        f"**Current period:** ${current_solved:,.0f} "
        f"({current_pct:.1f}% of income)"
    )
    if st.sidebar.button(
        "Use Current Period Target",
        help="Sets the real spending target to the solved amount for your currently selected start date and period.",
    ):
        st.session_state.apply_target = int(current_solved)
        st.rerun()

    worst_solved = st.session_state.solved_worst
    worst_pct = (worst_solved / annual_income) * 100
    worst_label = ""
    if st.session_state.get("solved_worst_start") is not None:
        ws = st.session_state.solved_worst_start
        wc = st.session_state.solved_worst_cagr
        worst_label = f"  \n{ws.strftime('%b %Y')} (CAGR {wc:.2%})"
    st.sidebar.warning(
        f"**Worst case:** ${worst_solved:,.0f} "
        f"({worst_pct:.1f}% of income){worst_label}"
    )
    if st.sidebar.button(
        "Use Worst Case Target",
        help="Sets the real spending target to the solved amount AND switches to the worst historical inflation period. "
             "This is the most conservative estimate — if your spending survives the worst case, it should survive any period.",
    ):
        st.session_state.apply_target = int(worst_solved)
        if st.session_state.get("solved_worst_start") is not None:
            ws = st.session_state.solved_worst_start
            st.session_state.apply_worst = (month_names[ws.month - 1], int(ws.year))
        st.rerun()

reserve_balance, actual_real_spending, real_spending_need_nominal, target_met = simulate_reserves(
    period, real_spending_target, reserve_spread
)

period["Real_Spending_Need"] = real_spending_need_nominal
period["Reserve_Balance"] = reserve_balance
period["Actual_Real_Spending"] = actual_real_spending
period["Target_Met"] = target_met

# --- Summary metrics ---
total_nominal = period["Nominal_Income"].sum()
total_real = period["Real_Income"].sum()
pp_lost = (1 - total_real / total_nominal) * 100

total_cumulative_inflation = (period["Cumulative_Inflation"].iloc[-1] - 1) * 100
n_years = len(period)
inflation_cagr = (period["Cumulative_Inflation"].iloc[-1] ** (1 / n_years) - 1) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Nominal Received", f"${total_nominal:,.0f}")
col2.metric("Total Real Value Received", f"${total_real:,.0f}")
col3.metric("Purchasing Power Lost", f"{pp_lost:.1f}%")
col4.metric("Total Inflation", f"{total_cumulative_inflation:.1f}%")
col5.metric("Inflation CAGR", f"{inflation_cagr:.2f}%")

first_label = period["Period_Start"].iloc[0].strftime("%b %Y")
last_label = period["Period_Start"].iloc[-1].strftime("%b %Y")
dollar_cost = period["Cumulative_Inflation"].iloc[-1]
st.markdown(
    f"**What cost \\$1.00 in {first_label} cost \\${dollar_cost:.2f} in {last_label}.**"
)

years_sustained = int(sum(period["Target_Met"]))
reserve_peak = period["Reserve_Balance"].max()
total_real_with_reserves = sum(period["Actual_Real_Spending"])
total_target = real_spending_target * len(period)
pct_of_target = (total_real_with_reserves / total_target) * 100 if total_target > 0 else 0

rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric("Years Target Sustained", f"{years_sustained} / {len(period)}")
rc2.metric("Reserve Peak Balance", f"${reserve_peak:,.0f}")
rc3.metric("Total Real Spending (w/ Reserves)", f"${total_real_with_reserves:,.0f}")
rc4.metric("% of Target Achieved", f"{pct_of_target:.1f}%")

# --- Chart ---
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=period["Period"], y=period["Nominal_Income"], name="Nominal Income", line=dict(color="#636EFA")
    )
)
fig.add_trace(
    go.Scatter(
        x=period["Period"], y=period["Real_Income"], name="Real Income (Unsmoothed)", line=dict(color="#EF553B")
    )
)
fig.add_trace(
    go.Scatter(
        x=period["Period"], y=period["Actual_Real_Spending"], name="Smoothed Real Income", line=dict(color="#00CC96", width=3)
    )
)
fig.update_layout(
    title="Nominal vs. Real Income Over Time",
    xaxis_title="Period",
    yaxis_title="Income ($)",
    yaxis_tickformat="$,.0f",
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# --- Reserve balance chart ---
fig_reserve = go.Figure()
fig_reserve.add_trace(
    go.Scatter(
        x=period["Period"], y=period["Reserve_Balance"], name="Reserve Balance",
        fill="tozeroy", line=dict(color="#AB63FA"),
    )
)
fig_reserve.update_layout(
    title="Reserve Fund Balance",
    xaxis_title="Period",
    yaxis_title="Balance ($)",
    yaxis_tickformat="$,.0f",
    height=350,
)
st.plotly_chart(fig_reserve, use_container_width=True)

# --- Data table ---
with st.expander("Year-by-Year Breakdown"):
    display = period[["Period", "Nominal_Income", "Annual_Inflation", "Cumulative_Inflation", "Real_Income",
                       "Real_Spending_Need", "Reserve_Balance", "Actual_Real_Spending"]].copy()
    display.columns = ["Period", "Nominal Income", "Annual Inflation", "Cumulative Inflation", "Real Income",
                        "Real Spending Need (Nominal)", "Reserve Balance", "Actual Real Spending"]
    display["Nominal Income"] = display["Nominal Income"].map("${:,.0f}".format)
    display["Annual Inflation"] = display["Annual Inflation"].map("{:.2%}".format)
    display["Cumulative Inflation"] = display["Cumulative Inflation"].map("{:.4f}".format)
    display["Real Income"] = display["Real Income"].map("${:,.0f}".format)
    display["Real Spending Need (Nominal)"] = display["Real Spending Need (Nominal)"].map("${:,.0f}".format)
    display["Reserve Balance"] = display["Reserve Balance"].map("${:,.0f}".format)
    display["Actual Real Spending"] = display["Actual Real Spending"].map("${:,.0f}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)
