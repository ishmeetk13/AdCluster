import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="AdCluster | Ad Performance Clustering",
    page_icon="🎯",
    layout="wide"
)

# Responsive CSS to fix metric card text clipping & support Light/Dark modes
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    
    /* Adapt metric cards dynamically with flexible padding */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 12px 14px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        overflow: hidden;
    }
    
    /* Prevent metric text clipping/ellipses */
    [data-testid="stMetricLabel"] {
        color: var(--text-color);
        opacity: 0.85;
        font-size: 0.85rem !important;
        white-space: normal !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-color);
        font-weight: 700;
        font-size: 1.3rem !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to transparently adapt Plotly charts to active Streamlit theme
def format_chart_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=None),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ----------------------------------------------------
# 2. LOAD CLUSTER MODEL BUNDLE
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    bundle = joblib.load('cluster_model.pkl')
    return bundle

try:
    bundle = load_assets()
    scaler = bundle['scaler']
    kmeans = bundle['kmeans']
    pca = bundle['pca']
    persona_map = bundle['persona_map']
    df_processed = bundle['df_processed']
    feature_cols = bundle['feature_cols']
except Exception as e:
    st.error(f"Error loading cluster_model.pkl: {e}")
    st.info("Ensure cluster_model.pkl is placed in the same directory as app.py.")
    st.stop()

# ----------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.title("🎯 AdCluster")
st.sidebar.caption("Ad Performance Clustering Model")
st.sidebar.markdown("---")

tab_selection = st.sidebar.radio(
    "Navigate",
    ["📊 Portfolio Audit", "🏷️ Persona Explorer", "🎯 New Ad Segmenter", "🌌 2D/3D Cluster Space"]
)

st.sidebar.markdown("---")
st.sidebar.info("Crisspo Performance Marketing Intelligence Layer")

# ----------------------------------------------------
# 4. TAB 1: PORTFOLIO AUDIT DASHBOARD
# ----------------------------------------------------
if tab_selection == "📊 Portfolio Audit":
    st.title("📊 Campaign Portfolio Audit")
    st.markdown("High-level breakdown of ad spend, conversion volume, and cluster persona distribution.")
    
    total_spend = df_processed['Spent'].sum()
    total_ads = len(df_processed)
    total_purchases = df_processed['Approved_Conversion'].sum()
    wasted_spend = df_processed[df_processed['Persona'] == 'Budget Drainers']['Spent'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Ads Analyzed", f"{total_ads:,}")
    with col2:
        st.metric("Total Campaign Spend", f"${total_spend:,.2f}")
    with col3:
        st.metric("Total Approved Sales", f"{total_purchases:,}")
    with col4:
        st.metric("Identified Wasted Spend", f"${wasted_spend:,.2f}", delta="-Pure Leakage", delta_color="inverse")
        
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Ad Volume by Persona")
        persona_counts = df_processed['Persona'].value_counts().reset_index()
        persona_counts.columns = ['Persona', 'Ad Count']
        
        fig_donut = px.pie(
            persona_counts, values='Ad Count', names='Persona',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(format_chart_theme(fig_donut), use_container_width=True)
        
    with c2:
        st.subheader("Budget Allocation vs. Wasted Spend")
        spend_by_persona = df_processed.groupby('Persona')['Spent'].sum().reset_index()
        
        fig_bar = px.bar(
            spend_by_persona, x='Spent', y='Persona',
            orientation='h',
            labels={'Spent': 'Total Spend ($)', 'Persona': 'Ad Persona'},
            color='Persona',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(format_chart_theme(fig_bar), use_container_width=True)

# ----------------------------------------------------
# 5. TAB 2: PERSONA EXPLORER
# ----------------------------------------------------
elif tab_selection == "🏷️ Persona Explorer":
    st.title("🏷️ Cluster Persona Profiles")
    st.markdown("Detailed breakdown of performance benchmarks and strategic actions per cluster.")
    
    summary_df = df_processed.groupby('Persona').agg(
        Avg_CTR=('CTR', 'mean'),
        Avg_CPC=('CPC', 'mean'),
        Avg_Conv_Rate=('Conv_Rate', 'mean'),
        Avg_CPP=('Cost_Per_Purchase', 'mean'),
        Avg_Spend=('Spent', 'mean'),
        Total_Spend=('Spent', 'sum'),
        Avg_Purchases=('Approved_Conversion', 'mean'),
        Total_Purchases=('Approved_Conversion', 'sum'),
        Ad_Count=('ad_id', 'count')
    ).reset_index()
    
    display_df = summary_df.copy()
    display_df['Avg CTR'] = display_df['Avg_CTR'].apply(lambda x: f"{x:.4%}")
    display_df['Avg CPC'] = display_df['Avg_CPC'].apply(lambda x: f"${x:.2f}")
    display_df['Avg Conv Rate'] = display_df['Avg_Conv_Rate'].apply(lambda x: f"{x:.2%}")
    display_df['Avg Cost/Purchase'] = display_df['Avg_CPP'].apply(lambda x: f"${x:.2f}")
    display_df['Avg Spend'] = display_df['Avg_Spend'].apply(lambda x: f"${x:.2f}")
    display_df['Total Spend'] = display_df['Total_Spend'].apply(lambda x: f"${x:,.2f}")
    display_df['Avg Purchases'] = display_df['Avg_Purchases'].apply(lambda x: f"{x:.2f}")
    display_df['Total Purchases'] = display_df['Total_Purchases'].apply(lambda x: f"{x:,.0f}")
    
    final_table = display_df[[
        'Persona', 'Ad_Count', 'Avg CTR', 'Avg CPC', 'Avg Conv Rate', 
        'Avg Cost/Purchase', 'Avg Spend', 'Total Spend', 
        'Avg Purchases', 'Total Purchases'
    ]].rename(columns={'Ad_Count': 'Ad Count'})
    
    st.dataframe(final_table, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Strategic Playbook per Persona")
    for cluster_id, details in persona_map.items():
        with st.expander(f"📌 **{details['name']}** — Recommended Action: `{details['action']}`"):
            st.write(f"**Profile Description:** {details['description']}")
            
            c_data = df_processed[df_processed['Cluster'] == cluster_id]
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Ad Count", len(c_data))
            with col_b:
                st.metric("Avg Conversion Rate", f"{c_data['Conv_Rate'].mean():.2%}")
            with col_c:
                st.metric("Avg Spend / Ad", f"${c_data['Spent'].mean():.2f}")

# ----------------------------------------------------
# 6. TAB 3: NEW AD SEGMENTER (SLIDER INPUTS CONSTRAINED TO MIN/MAX)
# ----------------------------------------------------
elif tab_selection == "🎯 New Ad Segmenter":
    st.title("🎯 Real-Time Ad Persona Segmenter")
    st.markdown("Use the sliders to set campaign performance metrics within observed historical ranges.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Ad Performance Inputs")
        
        # Pull min/max boundaries directly from training dataset
        min_imp, max_imp = int(df_processed['Impressions'].min()), int(df_processed['Impressions'].max())
        min_clicks, max_clicks = int(df_processed['Clicks'].min()), int(df_processed['Clicks'].max())
        min_spend_val, max_spend_val = float(df_processed['Spent'].min()), float(df_processed['Spent'].max())
        min_purch, max_purch = int(df_processed['Approved_Conversion'].min()), int(df_processed['Approved_Conversion'].max())
        
        # Sliders constrained strictly to min and max bounds
        impressions = st.slider("Impressions", min_value=min_imp, max_value=max_imp, value=10000, step=500)
        clicks = st.slider("Clicks", min_value=min_clicks, max_value=max_clicks, value=120, step=5)
        spent = st.slider("Total Spend ($)", min_value=min_spend_val, max_value=max_spend_val, value=50.0, step=1.0)
        purchases = st.slider("Approved Purchases", min_value=min_purch, max_value=max_purch, value=2, step=1)
        
        predict_btn = st.button("🚀 Analyze Ad Performance", type="primary", use_container_width=True)
        
    with col_result:
        st.subheader("Segmented Persona Output")
        
        if predict_btn:
            # 1. Ratios
            ctr = clicks / impressions if impressions > 0 else 0
            cpc = spent / clicks if clicks > 0 else 0
            conv_rate = purchases / clicks if clicks > 0 else 0
            
            # 2. Match exact training penalty logic
            if purchases > 0:
                cpp = spent / purchases
            else:
                cpp = spent + max_spend_val
            
            # 3. Create DataFrame and EXPLICITLY reorder to feature_cols
            raw_inputs = pd.DataFrame([{
                'CTR': ctr,
                'CPC': cpc,
                'Conv_Rate': conv_rate,
                'Cost_Per_Purchase': cpp,
                'Spent': spent
            }])[feature_cols]
            
            # 4. Scale & Predict
            scaled_inputs = scaler.transform(raw_inputs)
            predicted_cluster = int(kmeans.predict(scaled_inputs)[0])
            persona_info = persona_map[predicted_cluster]
            
            st.success(f"### Predicted Persona: **{persona_info['name']}**")
            st.info(f"**Recommended Action:** {persona_info['action']}")
            st.write(f"**Behavioral Profile:** {persona_info['description']}")
            
            st.markdown("---")
            st.caption("Calculated Ratios for Input:")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("CTR", f"{ctr:.3%}")
            with col_r2:
                st.metric("CPC", f"${cpc:.2f}")
            with col_r3:
                st.metric("Conv. Rate", f"{conv_rate:.2%}")
        else:
            st.info("👈 Adjust the sliders on the left and click **'Analyze Ad Performance'** to run the segmenter.")
# ----------------------------------------------------
# 7. TAB 4: 2D/3D CLUSTER SPACE VISUALIZER
# ----------------------------------------------------
elif tab_selection == "🌌 2D/3D Cluster Space":
    st.title("🌌 Cluster Geometry & PCA Visualizer")
    st.markdown("Explore how the 5 ad personas separate in multi-dimensional feature space.")
    
    viz_type = st.radio("Select Space Representation", ["2D PCA Projection", "3D Interactive Feature Space"], horizontal=True)
    
    if viz_type == "2D PCA Projection":
        st.subheader("2D Dimensionality Reduction (PCA)")
        fig_2d = px.scatter(
            df_processed,
            x='PCA1',
            y='PCA2',
            color='Persona',
            hover_data=['ad_id', 'Spent', 'Approved_Conversion', 'Conv_Rate'],
            labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
            title="Ad Clusters Projected onto 2D PCA Space",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(format_chart_theme(fig_2d), use_container_width=True)
        
    else:
        st.subheader("3D Feature Space: Spent vs. Conversion Rate vs. Cost Per Purchase")
        fig_3d = px.scatter_3d(
            df_processed,
            x='Spent',
            y='Conv_Rate',
            z='Cost_Per_Purchase',
            color='Persona',
            hover_data=['ad_id', 'Approved_Conversion'],
            labels={
                'Spent': 'Ad Spend ($)', 
                'Conv_Rate': 'Conversion Rate', 
                'Cost_Per_Purchase': 'Cost / Purchase ($)'
            },
            title="3D Ad Performance Clusters",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_3d.update_layout(scene=dict(zaxis=dict(range=[0, 1000])))
        st.plotly_chart(format_chart_theme(fig_3d), use_container_width=True)