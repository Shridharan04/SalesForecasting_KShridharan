import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# --- Sidebar navigation: lets you switch between pages ---
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

# @st.cache_data means: only actually re-read the CSV when the file changes,
# not on every single click
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Year'] = df['Order Date'].dt.year
    return df

df = load_data()

if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")

    # --- Total sales by year ---
    st.subheader("Total Sales by Year")
    yearly_sales = df.groupby('Year')['Sales'].sum()
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.bar(yearly_sales.index.astype(str), yearly_sales.values, color='#2563eb')
    ax1.set_ylabel('Sales ($)')
    st.pyplot(fig1)

    # --- Monthly sales trend ---
    st.subheader("Monthly Sales Trend")
    monthly_sales = df.set_index('Order Date').resample('MS')['Sales'].sum()
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(monthly_sales.index, monthly_sales.values, color='#16a34a')
    ax2.set_ylabel('Sales ($)')
    st.pyplot(fig2)

    # --- Interactive filters: sales by region and category ---
    st.subheader("Sales by Region & Category")
    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.selectbox("Filter by Region", ['All'] + sorted(df['Region'].unique().tolist()))
    with col2:
        selected_category = st.selectbox("Filter by Category", ['All'] + sorted(df['Category'].unique().tolist()))

    filtered = df.copy()
    if selected_region != 'All':
        filtered = filtered[filtered['Region'] == selected_region]
    if selected_category != 'All':
        filtered = filtered[filtered['Category'] == selected_category]

    st.write(f"Showing {len(filtered)} orders")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    filtered.groupby('Category')['Sales'].sum().plot(kind='bar', ax=ax3, color='#f59e0b')
    st.pyplot(fig3)

elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")

    @st.cache_data
    def load_forecasts():
        return pd.read_csv('forecast_results.csv', parse_dates=['Date'])

    fc_df = load_forecasts()

    col1, col2 = st.columns(2)
    with col1:
        seg_type = st.selectbox("View by", ["Category", "Region", "Total"])
    with col2:
        options = fc_df[fc_df['SegmentType'] == seg_type]['SegmentValue'].unique()
        seg_value = st.selectbox("Select", options)

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    selected = fc_df[(fc_df['SegmentType'] == seg_type) & (fc_df['SegmentValue'] == seg_value)].sort_values('Date').head(horizon)

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(selected['Date'], selected['Actual'], marker='o', label='Actual', color='black')
    ax.plot(selected['Date'], selected['Forecast'], marker='x', linestyle='--', label='SARIMA Forecast', color='#dc2626')
    ax.legend()
    ax.set_ylabel('Sales ($)')
    st.pyplot(fig)

    st.write(f"**MAE:** {selected['MAE'].iloc[0]:.1f}   **RMSE:** {selected['RMSE'].iloc[0]:.1f}")

elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")

    @st.cache_data
    def load_anomalies():
        return pd.read_csv('anomaly_results.csv', parse_dates=['Date'])

    an_df = load_anomalies()

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(an_df['Date'], an_df['Sales'], color='#94a3b8', linewidth=1, label='Weekly Sales')

    iso_only = an_df[an_df['IsoAnomaly'] & ~an_df['ZAnomaly']]
    z_only   = an_df[an_df['ZAnomaly'] & ~an_df['IsoAnomaly']]
    both     = an_df[an_df['IsoAnomaly'] & an_df['ZAnomaly']]

    ax.scatter(iso_only['Date'], iso_only['Sales'], color='red', s=60, label='Isolation Forest only', zorder=5)
    ax.scatter(z_only['Date'], z_only['Sales'], color='orange', s=60, label='Z-score only', zorder=5)
    ax.scatter(both['Date'], both['Sales'], color='purple', s=90, label='Both methods agree', zorder=6)
    ax.legend()
    ax.set_ylabel('Sales ($)')
    st.pyplot(fig)

    st.subheader("Detected Anomaly Weeks")
    anomaly_table = an_df[an_df['IsoAnomaly'] | an_df['ZAnomaly']][['Date','Sales','IsoAnomaly','ZAnomaly']]
    st.dataframe(anomaly_table)

elif page == "Product Demand Segments":
    st.title("📦 Product Demand Segments")

    @st.cache_data
    def load_clusters():
        return pd.read_csv('cluster_results.csv')

    cl_df = load_clusters()

    fig, ax = plt.subplots(figsize=(9,6))
    colors_map = {0:'#16a34a', 1:'#dc2626', 2:'#2563eb', 3:'#f59e0b'}
    for c in sorted(cl_df['Cluster'].unique()):
        subset = cl_df[cl_df['Cluster'] == c]
        ax.scatter(subset['PC1'], subset['PC2'], color=colors_map[c], s=120, label=subset['ClusterLabel'].iloc[0])
        for _, row in subset.iterrows():
            ax.annotate(row['Sub-Category'], (row['PC1'], row['PC2']), fontsize=8, xytext=(5,5), textcoords='offset points')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-Categories by Cluster")
    st.dataframe(cl_df[['Sub-Category', 'ClusterLabel', 'TotalSales', 'GrowthRate', 'Volatility', 'AvgOrderValue']]
                 .sort_values('ClusterLabel'))
