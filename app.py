import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gdown
import os
import zipfile
import warnings
from PIL import Image
from io import BytesIO
import requests

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="H&M Fashion BI - Emotion Analytics",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }
    .section-title {
        color: #2c3e50;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .tier-premium { background: linear-gradient(135deg, #1e5631 0%, #2d7a3e 100%); color: white; }
    .tier-trend { background: linear-gradient(135deg, #52b788 0%, #74c69d 100%); color: white; }
    .tier-stability { background: linear-gradient(135deg, #ffd60a 0%, #ffc300 100%); color: #333; }
    .tier-liquidation { background: linear-gradient(135deg, #ffb4a2 0%, #ff9999 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. FAST DATA ENGINE
# ============================================================================
@st.cache_resource
def init_resources():
    if not os.path.exists('data'):
        os.makedirs('data')
    
    files = {
        "data/article_master_web.csv": "1rLdTRGW2iu50edIDWnGSBkZqWznnNXLK",
        "data/customer_dna_master.csv": "182gmD8nYPAuy8JO_vIqzVJy8eMKqrGvH",
        "data/customer_test_validation.csv": "1mAufyQbOrpXdjkYXE4nhYyleGBoB6nXB",
        "data/visual_dna_embeddings.csv": "1VLNeGstZhn0_TdMiV-6nosxvxyFO5a54",
        "images.zip": "1z27fEDUpgXfiFzb1eUv5i5pbIA_cI7UA"
    }
    
    for path, fid in files.items():
        if not os.path.exists(path):
            try:
                gdown.download(f'https://drive.google.com/uc?id={fid}', path, quiet=False)
            except Exception as e:
                st.warning(f"⚠️ Could not download {path}: {str(e)}")
    
    if not os.path.exists('images') or len(os.listdir('images')) < 100:
        if not os.path.exists('images'):
            os.makedirs('images')
        try:
            if os.path.exists("images.zip"):
                with zipfile.ZipFile("images.zip", 'r') as z:
                    z.extractall('images')
        except Exception as e:
            st.warning(f"⚠️ Image extraction issue: {str(e)}")

@st.cache_data
def load_and_clean_data():
    df_a = pd.read_csv("data/article_master_web.csv")
    df_e = pd.read_csv("data/visual_dna_embeddings.csv")
    df_c = pd.read_csv("data/customer_dna_master.csv")
    
    df_a['article_id'] = df_a['article_id'].astype(str).str.zfill(10)
    df_e['article_id'] = df_e['article_id'].astype(str).str.zfill(10)
    
    return df_a, df_e, df_c

# Initialize
with st.spinner("🚀 Loading Strategic Engine..."):
    init_resources()
    df_art, df_emb, df_cust = load_and_clean_data()

# ============================================================================
# 3. NAVIGATION BAR (HORIZONTAL)
# ============================================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: white; text-align: center; margin: 0;">👗 H&M Fashion BI - Emotion-Driven Decision System</h1>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    if st.button("📊 Strategic Command", use_container_width=True):
        st.session_state.page = "strategic"
with col2:
    if st.button("🔍 Asset Optimization", use_container_width=True):
        st.session_state.page = "optimization"
with col3:
    if st.button("😊 Emotion DNA", use_container_width=True):
        st.session_state.page = "emotion"
with col4:
    if st.button("👥 Customer Behavior", use_container_width=True):
        st.session_state.page = "customer"
with col5:
    if st.button("🤖 AI Merchandising", use_container_width=True):
        st.session_state.page = "merchandising"
with col6:
    if st.button("📈 Financial Impact", use_container_width=True):
        st.session_state.page = "financial"

if "page" not in st.session_state:
    st.session_state.page = "strategic"

# ============================================================================
# 4. GLOBAL FILTERS
# ============================================================================
st.markdown("---")
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    emotions = ["All"] + sorted(df_art['mood'].dropna().unique().tolist())
    selected_emotion = st.selectbox("🎭 Filter by Emotion", emotions, key="emotion_filter")

with filter_col2:
    sections = ["All"] + sorted(df_art['section_name'].dropna().unique().tolist())
    selected_section = st.selectbox("🏪 Filter by Section", sections, key="section_filter")

with filter_col3:
    price_range = st.slider("💵 Price Range ($)", 
                            float(df_art['price'].min()), 
                            float(df_art['price'].max()), 
                            (float(df_art['price'].min()), float(df_art['price'].max())),
                            key="price_filter")

# Apply filters
filtered_df = df_art.copy()
if selected_emotion != "All":
    filtered_df = filtered_df[filtered_df['mood'] == selected_emotion]
if selected_section != "All":
    filtered_df = filtered_df[filtered_df['section_name'] == selected_section]
filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]

st.markdown("---")

# ============================================================================
# 5. PAGE 1: STRATEGIC COMMAND CENTER
# ============================================================================
if st.session_state.page == "strategic":
    st.markdown("<div class='section-title'>📊 Strategic Command Center</div>", unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📦 Total Products", len(filtered_df))
    with col2:
        st.metric("💰 Avg Price", f"${filtered_df['price'].mean():.2f}")
    with col3:
        st.metric("🔥 Avg Hotness", f"{filtered_df['hotness_score'].mean():.2f}")
    with col4:
        st.metric("👥 Emotions", filtered_df['mood'].nunique())
    with col5:
        st.metric("📊 Categories", filtered_df['section_name'].nunique())
    
    st.markdown("---")
    
    # Market Alignment Matrix
    st.subheader("📍 Market Alignment Matrix")
    emotion_stats = filtered_df.groupby('mood').agg({
        'hotness_score': 'mean',
        'price': 'mean',
        'article_id': 'count'
    }).reset_index()
    emotion_stats.columns = ['Emotion', 'Hotness', 'Price', 'Count']
    emotion_stats['Revenue'] = emotion_stats['Price'] * emotion_stats['Count']
    
    fig = px.scatter(emotion_stats, x='Price', y='Hotness', size='Revenue', 
                     color='Emotion', hover_name='Emotion', 
                     title="Price vs Hotness by Emotion",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Seasonality & Sentiment Drift
    st.subheader("📈 Emotion Distribution")
    emotion_dist = filtered_df['mood'].value_counts()
    fig = px.pie(values=emotion_dist.values, names=emotion_dist.index, 
                 title="Product Distribution by Emotion")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 6. PAGE 2: ASSET OPTIMIZATION & PRICING
# ============================================================================
elif st.session_state.page == "optimization":
    st.markdown("<div class='section-title'>🔍 Asset Optimization & Pricing Intelligence</div>", unsafe_allow_html=True)
    
    # Filters for this page
    st.subheader("🎯 Page Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        page_emotion = st.selectbox("Emotion", ["All"] + sorted(df_art['mood'].unique().tolist()), key="opt_emotion")
    with col2:
        page_category = st.selectbox("Category", ["All"] + sorted(df_art['section_name'].unique().tolist()), key="opt_category")
    with col3:
        page_group = st.selectbox("Product Group", ["All"] + sorted(df_art['product_group_name'].unique().tolist()), key="opt_group")
    
    # Apply page-specific filters
    page_filtered = filtered_df.copy()
    if page_emotion != "All":
        page_filtered = page_filtered[page_filtered['mood'] == page_emotion]
    if page_category != "All":
        page_filtered = page_filtered[page_filtered['section_name'] == page_category]
    if page_group != "All":
        page_filtered = page_filtered[page_filtered['product_group_name'] == page_group]
    
    st.markdown("---")
    
    # Two buttons for sections
    tab1, tab2 = st.tabs(["💰 4-Tier Pricing Strategy", "📊 Price Elasticity Simulator"])
    
    with tab1:
        st.subheader("💰 4-Tier Pricing Strategy")
        
        # Define tiers
        tier_premium = page_filtered[page_filtered['hotness_score'] > 0.8]
        tier_trend = page_filtered[(page_filtered['hotness_score'] >= 0.5) & (page_filtered['hotness_score'] <= 0.8)]
        tier_stability = page_filtered[(page_filtered['hotness_score'] >= 0.3) & (page_filtered['hotness_score'] < 0.5)]
        tier_liquidation = page_filtered[page_filtered['hotness_score'] < 0.3]
        
        # Display tier buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='tier-premium' style='padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>💎 Premium Tier</h3>
                <p>Hotness > 0.8</p>
                <p><b>{len(tier_premium)} Products</b></p>
                <p>Avg Price: ${tier_premium['price'].mean():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='tier-trend' style='padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>🔥 Trend Tier</h3>
                <p>Hotness 0.5-0.8</p>
                <p><b>{len(tier_trend)} Products</b></p>
                <p>Avg Price: ${tier_trend['price'].mean():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='tier-stability' style='padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>⚖️ Stability Tier</h3>
                <p>Hotness 0.3-0.5</p>
                <p><b>{len(tier_stability)} Products</b></p>
                <p>Avg Price: ${tier_stability['price'].mean():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='tier-liquidation' style='padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>📉 Liquidation Tier</h3>
                <p>Hotness < 0.3</p>
                <p><b>{len(tier_liquidation)} Products</b></p>
                <p>Avg Price: ${tier_liquidation['price'].mean():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tier selection
        selected_tier = st.radio("Select Tier to View Products", 
                                 ["Premium (>0.8)", "Trend (0.5-0.8)", "Stability (0.3-0.5)", "Liquidation (<0.3)"],
                                 horizontal=True)
        
        if selected_tier == "Premium (>0.8)":
            tier_data = tier_premium
        elif selected_tier == "Trend (0.5-0.8)":
            tier_data = tier_trend
        elif selected_tier == "Stability (0.3-0.5)":
            tier_data = tier_stability
        else:
            tier_data = tier_liquidation
        
        # Display products in grid
        st.subheader(f"📦 Products in {selected_tier}")
        cols = st.columns(5)
        for idx, (_, product) in enumerate(tier_data.head(20).iterrows()):
            with cols[idx % 5]:
                st.write(f"**{product['prod_name'][:20]}...**")
                st.write(f"Price: ${product['price']:.2f}")
                st.write(f"Hotness: {product['hotness_score']:.2f}")
                st.write(f"Emotion: {product['mood']}")
    
    with tab2:
        st.subheader("📊 Price Elasticity Simulator")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            price_increase = st.slider("Price Increase (%)", -30, 30, 0)
        with col2:
            demand_elasticity = st.slider("Demand Elasticity", -2.0, 0.0, -1.0)
        with col3:
            st.metric("Simulated Revenue Impact", f"{price_increase * demand_elasticity:.1f}%")
        
        # Simulation table
        simulation_data = page_filtered.head(10).copy()
        simulation_data['Original Price'] = simulation_data['price']
        simulation_data['New Price'] = simulation_data['price'] * (1 + price_increase/100)
        simulation_data['Revenue Impact'] = f"{price_increase * demand_elasticity:.1f}%"
        
        st.dataframe(simulation_data[['prod_name', 'Original Price', 'New Price', 'Revenue Impact', 'hotness_score']], 
                    use_container_width=True)

# ============================================================================
# 7. PAGE 3: EMOTIONAL PRODUCT DNA
# ============================================================================
elif st.session_state.page == "emotion":
    st.markdown("<div class='section-title'>😊 Emotional Product DNA</div>", unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Products", len(filtered_df))
    with col2:
        st.metric("💰 Avg Price", f"${filtered_df['price'].mean():.2f}")
    with col3:
        st.metric("🔥 Avg Hotness", f"{filtered_df['hotness_score'].mean():.2f}")
    with col4:
        st.metric("📊 Categories", filtered_df['section_name'].nunique())
    
    st.markdown("---")
    
    # Violin Plot
    st.subheader("📊 Price Distribution by Emotion")
    fig = px.violin(filtered_df, x='mood', y='price', box=True, points="outliers",
                   title="Price Distribution Across Emotions")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Sunburst Chart
    st.subheader("☀️ Category-Emotion Synergy")
    sunburst_data = filtered_df.groupby(['mood', 'section_name']).size().reset_index(name='count')
    
    fig = go.Figure(go.Sunburst(
        labels=['All'] + sunburst_data['mood'].unique().tolist() + sunburst_data['section_name'].unique().tolist(),
        parents=[''] + ['All']*len(sunburst_data['mood'].unique()) + sunburst_data['mood'].tolist(),
        values=[len(filtered_df)] + sunburst_data.groupby('mood')['count'].sum().tolist() + sunburst_data['count'].tolist(),
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Heroes Gallery
    st.subheader("⭐ Top Performing Heroes by Emotion")
    for emotion in filtered_df['mood'].unique():
        emotion_products = filtered_df[filtered_df['mood'] == emotion].nlargest(3, 'hotness_score')
        
        st.write(f"**{emotion}**")
        cols = st.columns(3)
        for idx, (_, product) in enumerate(emotion_products.iterrows()):
            with cols[idx]:
                st.write(f"🏆 {product['prod_name'][:30]}")
                st.write(f"Hotness: {product['hotness_score']:.2f}")
                st.write(f"Price: ${product['price']:.2f}")

# ============================================================================
# 8. PAGE 4: CUSTOMER SEGMENTATION & BEHAVIOR
# ============================================================================
elif st.session_state.page == "customer":
    st.markdown("<div class='section-title'>👥 Customer Segmentation & Behavior</div>", unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Customers", len(df_cust))
    with col2:
        st.metric("📅 Avg Age", f"{df_cust['age'].mean():.0f}")
    with col3:
        st.metric("💰 Avg Spending", f"${df_cust['avg_spending'].mean():.2f}")
    with col4:
        st.metric("🛍️ Avg Purchases", f"{df_cust['purchase_count'].mean():.0f}")
    
    st.markdown("---")
    
    # Segment Distribution
    st.subheader("📊 Customer Segment Distribution")
    segment_dist = df_cust['segment'].value_counts()
    fig = px.pie(values=segment_dist.values, names=segment_dist.index,
                title="Customer Segments")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Age-Price Sensitivity
    st.subheader("💰 Spending vs Age by Segment")
    fig = px.scatter(df_cust, x='age', y='avg_spending', color='segment',
                    size='purchase_count', hover_name='customer_id',
                    title="Customer Age vs Spending Pattern")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Loyalists
    st.subheader("⭐ Top Loyalists")
    top_customers = df_cust.nlargest(10, 'purchase_count')[['customer_id', 'age', 'segment', 'avg_spending', 'purchase_count']]
    st.dataframe(top_customers, use_container_width=True)
    
    st.markdown("---")
    
    # Customer Persona Cards
    st.subheader("👤 Customer Persona Insights")
    
    personas = {
        'Gold': df_cust[df_cust['segment'] == 'Gold'],
        'Silver': df_cust[df_cust['segment'] == 'Silver'],
        'Bronze': df_cust[df_cust['segment'] == 'Bronze']
    }
    
    cols = st.columns(3)
    for idx, (segment, data) in enumerate(personas.items()):
        with cols[idx]:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; color: white;'>
                <h3>{segment} Segment</h3>
                <p><b>Count:</b> {len(data)}</p>
                <p><b>Avg Age:</b> {data['age'].mean():.0f}</p>
                <p><b>Avg Spending:</b> ${data['avg_spending'].mean():.2f}</p>
                <p><b>Avg Purchases:</b> {data['purchase_count'].mean():.0f}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# 9. PAGE 5: AI VISUAL MERCHANDISING
# ============================================================================
elif st.session_state.page == "merchandising":
    st.markdown("<div class='section-title'>🤖 AI Visual Merchandising</div>", unsafe_allow_html=True)
    
    # Filters
    st.subheader("🎯 Page Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        merch_emotion = st.selectbox("Emotion", ["All"] + sorted(df_art['mood'].unique().tolist()), key="merch_emotion")
    with col2:
        merch_category = st.selectbox("Category", ["All"] + sorted(df_art['section_name'].unique().tolist()), key="merch_category")
    with col3:
        merch_group = st.selectbox("Product Group", ["All"] + sorted(df_art['product_group_name'].unique().tolist()), key="merch_group")
    with col4:
        merch_price = st.slider("Price Range", float(df_art['price'].min()), float(df_art['price'].max()), 
                               (float(df_art['price'].min()), float(df_art['price'].max())), key="merch_price")
    
    # Apply filters
    merch_filtered = filtered_df.copy()
    if merch_emotion != "All":
        merch_filtered = merch_filtered[merch_filtered['mood'] == merch_emotion]
    if merch_category != "All":
        merch_filtered = merch_filtered[merch_filtered['section_name'] == merch_category]
    if merch_group != "All":
        merch_filtered = merch_filtered[merch_filtered['product_group_name'] == merch_group]
    merch_filtered = merch_filtered[(merch_filtered['price'] >= merch_price[0]) & (merch_filtered['price'] <= merch_price[1])]
    
    st.markdown("---")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📦 Products", len(merch_filtered))
    with col2:
        st.metric("💰 Avg Price", f"${merch_filtered['price'].mean():.2f}")
    with col3:
        st.metric("🔥 Avg Hotness", f"{merch_filtered['hotness_score'].mean():.2f}")
    with col4:
        high_performers = len(merch_filtered[merch_filtered['hotness_score'] > 0.7])
        st.metric("⭐ High Performers", high_performers)
    with col5:
        revenue_potential = (merch_filtered['price'] * merch_filtered['hotness_score']).sum()
        st.metric("💵 Revenue Potential", f"${revenue_potential:.0f}")
    
    st.markdown("---")
    
    # Neural Similarity Engine
    st.subheader("🧠 Neural Similarity Engine")
    
    selected_product = st.selectbox("Select a Product", 
                                   merch_filtered['prod_name'].values,
                                   key="neural_product")
    
    selected_product_data = merch_filtered[merch_filtered['prod_name'] == selected_product].iloc[0]
    
    # Find similar products
    similar_products = merch_filtered[
        (merch_filtered['mood'] == selected_product_data['mood']) &
        (merch_filtered['article_id'] != selected_product_data['article_id'])
    ].nlargest(10, 'hotness_score')
    
    st.markdown("---")
    
    # Smart Match Engine
    st.subheader("🎯 Smart Match Engine - Top Similar Products")
    
    cols = st.columns(5)
    for idx, (_, product) in enumerate(similar_products.iterrows()):
        with cols[idx % 5]:
            st.write(f"**{product['prod_name'][:20]}...**")
            st.write(f"Price: ${product['price']:.2f}")
            st.write(f"Hotness: {product['hotness_score']:.2f}")
            st.write(f"Emotion: {product['mood']}")
            if st.button("View Details", key=f"detail_{product['article_id']}"):
                st.session_state.selected_product_detail = product
    
    # Product Detail Display
    if "selected_product_detail" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Product Details")
        product = st.session_state.selected_product_detail
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Product Name:** {product['prod_name']}")
            st.write(f"**Category:** {product['section_name']}")
            st.write(f"**Group:** {product['product_group_name']}")
            st.write(f"**Price:** ${product['price']:.2f}")
        with col2:
            st.write(f"**Hotness Score:** {product['hotness_score']:.2f}")
            st.write(f"**Emotion:** {product['mood']}")
            st.write(f"**Description:** {product.get('detail_desc', 'N/A')[:100]}...")
    
    st.markdown("---")
    
    # Match Score Analytics
    st.subheader("📡 Match Score Analytics")
    
    match_scores = []
    for _, product in similar_products.head(5).iterrows():
        emotion_match = 1.0 if product['mood'] == selected_product_data['mood'] else 0.5
        price_match = 1.0 - abs(product['price'] - selected_product_data['price']) / selected_product_data['price']
        hotness_match = product['hotness_score'] / 1.0
        overall_match = (emotion_match * 0.4 + price_match * 0.3 + hotness_match * 0.3)
        match_scores.append({
            'Product': product['prod_name'][:20],
            'Overall': overall_match,
            'Emotion': emotion_match,
            'Price': price_match,
            'Hotness': hotness_match
        })
    
    match_df = pd.DataFrame(match_scores)
    
    fig = go.Figure(data=[
        go.Scatterpolar(r=match_df['Overall'], theta=match_df['Product'], fill='toself', name='Overall Match'),
    ])
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 10. PAGE 6: FINANCIAL IMPACT & PERFORMANCE
# ============================================================================
elif st.session_state.page == "financial":
    st.markdown("<div class='section-title'>📈 Financial Impact & Performance</div>", unsafe_allow_html=True)
    
    # Filters
    st.subheader("🎯 Page Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fin_emotion = st.selectbox("Emotion", ["All"] + sorted(df_art['mood'].unique().tolist()), key="fin_emotion")
    with col2:
        fin_category = st.selectbox("Category", ["All"] + sorted(df_art['section_name'].unique().tolist()), key="fin_category")
    with col3:
        fin_group = st.selectbox("Product Group", ["All"] + sorted(df_art['product_group_name'].unique().tolist()), key="fin_group")
    with col4:
        fin_price = st.slider("Price Range", float(df_art['price'].min()), float(df_art['price'].max()), 
                             (float(df_art['price'].min()), float(df_art['price'].max())), key="fin_price")
    
    # Apply filters
    fin_filtered = filtered_df.copy()
    if fin_emotion != "All":
        fin_filtered = fin_filtered[fin_filtered['mood'] == fin_emotion]
    if fin_category != "All":
        fin_filtered = fin_filtered[fin_filtered['section_name'] == fin_category]
    if fin_group != "All":
        fin_filtered = fin_filtered[fin_filtered['product_group_name'] == fin_group]
    fin_filtered = fin_filtered[(fin_filtered['price'] >= fin_price[0]) & (fin_filtered['price'] <= fin_price[1])]
    
    st.markdown("---")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        revenue_potential = (fin_filtered['price'] * fin_filtered['hotness_score']).sum()
        st.metric("💰 Revenue Potential", f"${revenue_potential:.0f}")
    with col2:
        avg_margin = fin_filtered['hotness_score'].mean() * 100
        st.metric("📊 Avg Margin %", f"{avg_margin:.1f}%")
    with col3:
        high_performers = len(fin_filtered[fin_filtered['hotness_score'] > 0.6])
        st.metric("⭐ High Performers", high_performers)
    with col4:
        low_performers = len(fin_filtered[fin_filtered['hotness_score'] < 0.4])
        st.metric("📉 Low Performers", low_performers)
    
    st.markdown("---")
    
    # Revenue by Category
    st.subheader("📊 Revenue by Category")
    category_revenue = fin_filtered.groupby('section_name').agg({
        'price': 'sum',
        'hotness_score': 'mean'
    }).reset_index()
    category_revenue.columns = ['Category', 'Revenue', 'Avg Hotness']
    
    fig = px.bar(category_revenue, x='Category', y='Revenue', color='Avg Hotness',
                title="Revenue Potential by Category")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Hotness Performance
    st.subheader("🔥 Hotness Performance Distribution")
    fig = px.histogram(fin_filtered, x='hotness_score', nbins=20,
                      title="Product Hotness Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Investment Strategy
    st.subheader("📋 Investment Strategy - Invest vs Divest")
    
    invest_products = fin_filtered[fin_filtered['hotness_score'] > 0.6]
    maintain_products = fin_filtered[(fin_filtered['hotness_score'] >= 0.4) & (fin_filtered['hotness_score'] <= 0.6)]
    divest_products = fin_filtered[fin_filtered['hotness_score'] < 0.4]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='background: #1e5631; padding: 20px; border-radius: 10px; color: white;'>
            <h3>💎 INVEST</h3>
            <p>High Hotness (>0.6)</p>
            <p><b>{len(invest_products)} Products</b></p>
            <p>Increase inventory & marketing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: #ffc300; padding: 20px; border-radius: 10px; color: #333;'>
            <h3>⚖️ MAINTAIN</h3>
            <p>Medium Hotness (0.4-0.6)</p>
            <p><b>{len(maintain_products)} Products</b></p>
            <p>Keep current levels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: #ff6b6b; padding: 20px; border-radius: 10px; color: white;'>
            <h3>📉 DIVEST</h3>
            <p>Low Hotness (<0.4)</p>
            <p><b>{len(divest_products)} Products</b></p>
            <p>Reduce inventory & reposition</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Profit Recovery Tracker
    st.subheader("💰 Profit Recovery Tracker")
    
    base_profit = 121
    recovered_profit = 151
    recovery_gain = recovered_profit - base_profit
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Base Profit", f"${base_profit}")
    with col2:
        st.metric("Recovered Profit", f"${recovered_profit}")
    with col3:
        st.metric("Recovery Gain", f"${recovery_gain} ({recovery_gain/base_profit*100:.1f}%)")
    
    st.markdown(f"""
    <div class='insight-box'>
        <h4>💡 Profit Recovery Recommendation</h4>
        <p>By implementing the investment strategy above, you can recover <b>${recovery_gain}</b> in profit.</p>
        <p>This represents a <b>{recovery_gain/base_profit*100:.1f}%</b> improvement over base profit.</p>
        <p><b>Action Items:</b></p>
        <ul>
            <li>Increase inventory for {len(invest_products)} high-performing products</li>
            <li>Maintain current strategy for {len(maintain_products)} stable products</li>
            <li>Reduce or reposition {len(divest_products)} underperforming products</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #999;'>H&M Fashion BI | Emotion-Driven Decision System | © 2024</p>", unsafe_allow_html=True)
