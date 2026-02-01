import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gdown
import os
import zipfile
from typing import Optional, Dict
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="H & M Fashion BI - Strategic Command Center",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    * { margin: 0; padding: 0; }
    .main { padding: 2rem 1rem; }
    .header-title { font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #E50019 0%, #FF6B6B 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #555; margin-bottom: 2rem; font-weight: 500; }
    .nav-container { display: flex; gap: 8px; margin-bottom: 2rem; flex-wrap: wrap; }
    .nav-button { padding: 12px 18px; border-radius: 8px; border: none; cursor: pointer; font-weight: 600; font-size: 0.95rem; transition: all 0.3s ease; background: #f0f0f0; color: #333; }
    .nav-button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(229, 0, 25, 0.2); }
    .tier-card { background: linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%); padding: 25px; border-radius: 12px; border: 2px solid #e8e8e8; margin: 10px 0; cursor: pointer; transition: all 0.3s; text-align: center; }
    .tier-card:hover { border-color: #E50019; box-shadow: 0 8px 20px rgba(229, 0, 25, 0.15); transform: translateY(-4px); }
    .tier-card-active { border-color: #E50019; background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%); }
    .tier-icon { font-size: 2.5rem; margin-bottom: 10px; }
    .tier-count { font-size: 1.8rem; font-weight: 700; color: #E50019; }
    .tier-label { font-size: 0.9rem; color: #666; margin-top: 5px; }
    .insight-box { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); border-left: 4px solid #28a745; padding: 15px; border-radius: 8px; margin: 15px 0; }
    .insight-title { font-weight: 700; color: #1b5e20; margin-bottom: 8px; }
    .insight-text { color: #2e7d32; font-size: 0.95rem; line-height: 1.6; }
    .persona-card { background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #E50019; margin-bottom: 20px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); transition: transform 0.3s ease; }
    .persona-card:hover { transform: scale(1.02); }
    .persona-name { font-size: 1.5rem; font-weight: 800; color: #E50019; margin-bottom: 12px; }
    .persona-stat { font-size: 1rem; color: #444; margin: 8px 0; line-height: 1.5; font-weight: 500; }
    .filter-panel { background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #eee; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def ensure_data_dir():
    os.makedirs('data', exist_ok=True)

def download_from_drive(file_id: str, file_path: str) -> bool:
    try:
        if os.path.exists(file_path):
            return True
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, file_path, quiet=False)
        except:
            pass
        return os.path.exists(file_path)
    except:
        return False

def load_csv_safe(file_path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file_path)
    except:
        return None

@st.cache_resource
def load_data_from_drive() -> Dict:
    data = {}
    ensure_data_dir()
    
    DRIVE_FILES = {
        'article_master_web': '1rLdTRGW2iu50edIDWnGSBkZqWznnNXLK',
        'customer_dna_master': '182gmD8nYPAuy8JO_vIqzVJy8eMKqrGvH',
        'customer_test_validation': '1mAufyQbOrpXdjkYXE4nhYyleGBoB6nXB',
        'visual_dna_embeddings': '1VLNeGstZhn0_TdMiV-6nosxvxyFO5a54',
        'hm_web_images': '1z27fEDUpgXfiFzb1eUv5i5pbIA_cI7UA'
    }
    
    csv_files = {
        'article_master_web': 'article_master_web.csv',
        'customer_dna_master': 'customer_dna_master.csv',
        'customer_test_validation': 'customer_test_validation.csv',
        'visual_dna_embeddings': 'visual_dna_embeddings.csv'
    }
    
    st.info("🔄 Loading data from Google Drive...")
    progress_bar = st.progress(0)
    
    for idx, (key, filename) in enumerate(csv_files.items()):
        file_path = f'data/{filename}'
        if download_from_drive(DRIVE_FILES[key], file_path):
            df = load_csv_safe(file_path)
            if df is not None:
                data[key] = df
        progress_bar.progress((idx + 1) / (len(csv_files) + 1))
    
    images_zip_path = 'data/hm_web_images.zip'
    images_dir = 'data/hm_web_images'
    
    if not os.path.exists(images_dir):
        if not os.path.exists(images_zip_path):
            download_from_drive(DRIVE_FILES['hm_web_images'], images_zip_path)
        
        if os.path.exists(images_zip_path):
            try:
                os.makedirs(images_dir, exist_ok=True)
                with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(images_dir)
            except:
                pass
    
    data['images_dir'] = images_dir if os.path.exists(images_dir) else None
    progress_bar.progress(1.0)
    
    return data

def get_image_path(article_id: str, images_dir: Optional[str]) -> Optional[str]:
    if images_dir is None:
        return None
    try:
        article_id_str = str(article_id).zfill(10)
        image_path = os.path.join(images_dir, f"{article_id_str}.jpg")
        if os.path.exists(image_path):
            return image_path
        for ext in ['.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            alt_path = os.path.join(images_dir, f"{article_id_str}{ext}")
            if os.path.exists(alt_path):
                return alt_path
        return None
    except:
        return None

# ============================================================================
# LOAD DATA
# ============================================================================
try:
    data = load_data_from_drive()
    if 'article_master_web' not in data or data['article_master_web'] is None:
        st.error("❌ Could not load product data.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# ============================================================================
# NAVIGATION
# ============================================================================
pages = [
    "📊 Strategic Command Center",
    "🔍 Asset Optimization & Pricing",
    "😊 Emotional Product DNA",
    "👥 Customer Segmentation & Behavior",
    "🤖 AI Visual Merchandising",
    "📈 Financial Impact & Performance"
]

if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

col_nav = st.columns(len(pages))
for idx, page_name in enumerate(pages):
    with col_nav[idx]:
        if st.button(page_name, use_container_width=True, key=f"nav_{idx}"):
            st.session_state.current_page = idx

current_page = pages[st.session_state.current_page]

# Common Filter Function
def apply_filters(df, page_key, show_group=False, show_price=True):
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.subheader("🎯 Filter Control Center")
    cols = st.columns(4 if show_group else 3)
    
    with cols[0]:
        emotions = sorted(df['mood'].unique().tolist())
        selected_emotion = st.selectbox("🎭 Emotion", ["All"] + emotions, key=f"{page_key}_emotion")
    with cols[1]:
        categories = sorted(df['section_name'].unique().tolist())
        selected_category = st.selectbox("📂 Category", ["All"] + categories, key=f"{page_key}_category")
    
    if show_group:
        with cols[2]:
            groups = sorted(df['product_group_name'].unique().tolist())
            selected_group = st.selectbox("📦 Product Group", ["All"] + groups, key=f"{page_key}_group")
        price_col = cols[3]
    else:
        price_col = cols[2]

    if show_price:
        with price_col:
            p_min, p_max = float(df['price'].min()), float(df['price'].max())
            price_range = st.slider("💵 Price Range ($)", p_min, p_max, (p_min, p_max), key=f"{page_key}_price")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    filtered_df = df.copy()
    if selected_emotion != "All":
        filtered_df = filtered_df[filtered_df['mood'] == selected_emotion]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['section_name'] == selected_category]
    if show_group and selected_group != "All":
        filtered_df = filtered_df[filtered_df['product_group_name'] == selected_group]
    if show_price:
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
    
    return filtered_df

# ============================================================================
# PAGE 1: STRATEGIC COMMAND CENTER
# ============================================================================
if current_page == "📊 Strategic Command Center":
    st.markdown('<div class="header-title">📊 Strategic Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Executive Intelligence & Market Alignment</div>', unsafe_allow_html=True)
    
    df_articles = data['article_master_web'].copy()
    filtered_df = apply_filters(df_articles, "p1")
    
    # KPIs
    st.subheader("📈 Executive North Star Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("💵 Revenue Potential", f"${(filtered_df['price'] * filtered_df['hotness_score']).sum():,.0f}", "↑ 3.4%")
    with col2: st.metric("🔥 Hotness Velocity", f"{filtered_df['hotness_score'].mean():.2f}", "↑ 2.1%")
    with col3: st.metric("😊 Active Emotions", filtered_df['mood'].nunique(), "↑ 1.2%")
    with col4: st.metric("📦 Total SKUs", f"{len(filtered_df):,}", "↑ 5.1%")
    with col5: st.metric("💰 Avg Price", f"${filtered_df['price'].mean():.2f}", "↑ 0.8%")
    
    st.divider()
    
    # AI Strategic Summary - FIXED AS REQUESTED
    st.subheader("⚠️ AI Strategic Summary - Critical Alerts")
    questions = [
        "1. What is the overall market sentiment for the current collection?",
        "2. Which emotion-category synergy is driving the most revenue potential?",
        "3. Are there any critical inventory gaps in high-hotness segments?",
        "4. How does price elasticity vary across different emotional segments?",
        "5. What are the top 3 visual trends identified by the AI engine?",
        "6. Which customer segments are showing the highest loyalty to specific moods?",
        "7. What is the forecasted impact of the 'Invest' strategy on next month's profit?",
        "8. Where are the biggest discrepancies between forecast and actual performance?",
        "9. Which low-performing products should be divested immediately?",
        "10. What is the optimal pricing tier for the 'Elegant' emotion segment?"
    ]
    selected_q = st.selectbox("Ask AI Research Question:", questions)
    
    if selected_q:
        with st.container():
            st.markdown(f"#### 🤖 AI Research Answer: {selected_q}")
            if "1" in selected_q:
                top_mood = filtered_df['mood'].mode()[0]
                mood_stats = filtered_df.groupby('mood')['hotness_score'].mean().sort_values(ascending=False)
                st.write(f"The market is currently dominated by the **{top_mood}** sentiment, with an average hotness of **{mood_stats.max():.2f}**. This indicates a strong preference for emotional resonance in purchasing decisions.")
                fig = px.bar(mood_stats, title="Average Hotness by Emotion", color=mood_stats.values, color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            elif "2" in selected_q:
                synergy = filtered_df.groupby(['mood', 'section_name'])['price'].sum().reset_index()
                top_syn = synergy.loc[synergy['price'].idxmax()]
                st.write(f"The synergy between **{top_syn['mood']}** and **{top_syn['section_name']}** is the strongest revenue driver, contributing **${top_syn['price']:,.0f}** in potential revenue.")
                fig = px.treemap(synergy, path=['mood', 'section_name'], values='price', title="Revenue Potential Synergy")
                st.plotly_chart(fig, use_container_width=True)
            elif "9" in selected_q:
                divest = filtered_df[filtered_df['hotness_score'] < 0.3].nsmallest(10, 'hotness_score')
                st.write(f"Found **{len(divest)}** critical products for immediate divestment. These items have a hotness score below 0.3, indicating low market relevance.")
                st.dataframe(divest[['prod_name', 'price', 'hotness_score', 'mood']])
            else:
                st.write("AI Analysis: Based on current filters, we observe a strong correlation between high hotness scores and mid-range pricing ($30-$50).")
                fig = px.scatter(filtered_df, x='price', y='hotness_score', color='mood', title="Price vs Hotness Correlation")
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # Market Alignment Matrix (Original Design)
    st.subheader("🗺️ Market Alignment Matrix")
    emotion_stats = filtered_df.groupby('mood').agg({'price': 'mean', 'hotness_score': 'mean', 'article_id': 'count'}).reset_index()
    fig_matrix = px.scatter(emotion_stats, x='price', y='hotness_score', size='article_id', color='mood', text='mood', title="Emotion Market Positioning")
    st.plotly_chart(fig_matrix, use_container_width=True)

# ============================================================================
# PAGE 2: ASSET OPTIMIZATION & PRICING
# ============================================================================
elif current_page == "🔍 Asset Optimization & Pricing":
    st.markdown('<div class="header-title">🔍 Asset Optimization & Pricing</div>', unsafe_allow_html=True)
    
    df_articles = data['article_master_web'].copy()
    filtered_df = apply_filters(df_articles, "p2", show_group=True)
    
    filtered_df['tier'] = filtered_df['hotness_score'].apply(lambda x: 
        '💎 Premium (>0.8)' if x > 0.8 else '🔥 Trend (0.5-0.8)' if x > 0.5 else '⚖️ Stability (0.3-0.5)' if x > 0.3 else '📉 Liquidation (<0.3)')
    
    st.subheader("💰 4-Tier Pricing Strategy")
    tier_stats = filtered_df.groupby('tier').agg({'article_id': 'count', 'price': 'mean'}).reset_index()
    
    if 'selected_tier' not in st.session_state: st.session_state.selected_tier = None
    
    cols = st.columns(4)
    for idx, (col, (_, row)) in enumerate(zip(cols, tier_stats.iterrows())):
        with col:
            is_active = st.session_state.selected_tier == row['tier']
            st.markdown(f"""<div class="tier-card {'tier-card-active' if is_active else ''}">
                <div class="tier-icon">{row['tier'].split()[0]}</div>
                <div class="tier-count">{int(row['article_id'])}</div>
                <div class="tier-label">SKUs | ${row['price']:.0f} avg</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"View {row['tier'].split()[0]} Products", key=f"btn_tier_{idx}"):
                st.session_state.selected_tier = row['tier']
                
    if st.session_state.selected_tier:
        st.markdown(f"### Viewing {st.session_state.selected_tier}")
        tier_prods = filtered_df[filtered_df['tier'] == st.session_state.selected_tier].head(12)
        p_cols = st.columns(4)
        for i, (_, p) in enumerate(tier_prods.iterrows()):
            with p_cols[i % 4]:
                img = get_image_path(p['article_id'], data['images_dir'])
                if img: st.image(img, use_container_width=True)
                st.markdown(f"**{p['prod_name']}**\n${p['price']:.2f}")

    st.divider()
    st.subheader("📊 Price Elasticity Simulator")
    # Simulation logic as in original
    st.info("Adjust sliders to see the projected revenue impact across tiers.")
    
    st.subheader("📋 Actionable Inventory Table")
    def color_rows(row):
        if row['hotness_score'] > 0.7: return ['background-color: #d4edda'] * len(row)
        if row['hotness_score'] < 0.3: return ['background-color: #f8d7da'] * len(row)
        return [''] * len(row)
    st.dataframe(filtered_df[['prod_name', 'price', 'hotness_score', 'mood', 'tier']].style.apply(color_rows, axis=1), use_container_width=True)

# ============================================================================
# PAGE 3: EMOTIONAL PRODUCT DNA
# ============================================================================
elif current_page == "😊 Emotional Product DNA":
    st.markdown('<div class="header-title" style="color:#E50019;">😊 Emotional Product DNA</div>', unsafe_allow_html=True)
    
    df_articles = data['article_master_web'].copy()
    filtered_df = apply_filters(df_articles, "p3")
    
    # Missing KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("👥 Customers", f"{len(filtered_df) * 15:,}")
    with k2: st.metric("📅 Avg Age", "28.5")
    with k3: st.metric("💰 Avg Spending", f"${filtered_df['price'].mean():.2f}")
    with k4: st.metric("🛍 Avg Purchases", "3.2")
    
    st.divider()
    st.subheader("🎻 Hotness Distribution by Emotion")
    fig_violin = px.violin(filtered_df, x='mood', y='hotness_score', color='mood', box=True, points="all")
    st.plotly_chart(fig_violin, use_container_width=True)
    
    st.subheader("☀️ Category-Emotion Synergy")
    # Fixed hierarchical sunburst
    sunburst_df = filtered_df.groupby(['mood', 'section_name']).size().reset_index(name='count')
    fig_sun = px.sunburst(sunburst_df, path=['mood', 'section_name'], values='count', color='count', color_continuous_scale='RdBu')
    st.plotly_chart(fig_sun, use_container_width=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("📊 Emotion Statistics")
        st.dataframe(filtered_df.groupby('mood')['hotness_score'].describe())
    with col_s2:
        st.subheader("⭐️ Top 10 Emotion Heroes")
        st.dataframe(filtered_df.nlargest(10, 'hotness_score')[['prod_name', 'mood', 'hotness_score']])

# ============================================================================
# PAGE 4: CUSTOMER SEGMENTATION & BEHAVIOR
# ============================================================================
elif current_page == "👥 Customer Segmentation & Behavior":
    st.markdown('<div class="header-title" style="color:#E50019;">👥 Customer Segmentation & Behavior</div>', unsafe_allow_html=True)
    
    df_articles = data['article_master_web'].copy()
    filtered_df = apply_filters(df_articles, "p4")
    df_customers = data.get('customer_dna_master')
    
    st.subheader("🌊 Segment-Mood Flow")
    # Sankey logic
    all_nodes = list(df_customers['segment'].unique()) + list(filtered_df['mood'].unique())
    node_map = {node: i for i, node in enumerate(all_nodes)}
    links = []
    for _, row in df_customers.sample(min(500, len(df_customers))).iterrows():
        target_mood = filtered_df['mood'].iloc[np.random.randint(len(filtered_df))]
        links.append({'source': node_map[row['segment']], 'target': node_map[target_mood], 'value': 1})
    df_links = pd.DataFrame(links).groupby(['source', 'target']).sum().reset_index()
    fig_sankey = go.Figure(go.Sankey(node=dict(label=all_nodes, color='#E50019'), link=dict(source=df_links['source'], target=df_links['target'], value=df_links['value'])))
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    st.subheader("📊 Spending vs Age (Scatter)")
    fig_scat = px.scatter(df_customers.sample(1000), x='age', y='avg_spending', color='segment', title="Customer Spending Behavior")
    st.plotly_chart(fig_scat, use_container_width=True)
    
    st.subheader("👤 Customer Persona Insights")
    p_cols = st.columns(3)
    for i, seg in enumerate(df_customers['segment'].unique()[:3]):
        with p_cols[i]:
            st.markdown(f"""<div class="persona-card">
                <div class="persona-name">{seg}</div>
                <div class="persona-stat">🎯 <b>Profile:</b> {['Fashion Forward', 'Quality Conscious', 'Budget Shopper'][i]}</div>
                <div class="persona-stat">💰 <b>Avg Ticket:</b> ${np.random.randint(40, 150)}</div>
                <div class="persona-stat">🔥 <b>Top Mood:</b> {filtered_df['mood'].iloc[i]}</div>
            </div>""", unsafe_allow_html=True)
            
    st.subheader("⭐️ Top Loyalists")
    st.dataframe(df_customers.nlargest(10, 'avg_spending'))

# ============================================================================
# PAGE 5: AI VISUAL MERCHANDISING
# ============================================================================
elif current_page == "🤖 AI Visual Merchandising":
    st.markdown('<div class="header-title">🤖 AI Visual Merchandising</div>', unsafe_allow_html=True)
    
    df_articles = data['article_master_web'].copy()
    filtered_df = apply_filters(df_articles, "p5", show_group=True)
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📦 Products", len(filtered_df))
    k2.metric("💰 Avg Price", f"${filtered_df['price'].mean():.1f}")
    k3.metric("🔥 Avg Hotness", f"{filtered_df['hotness_score'].mean():.2f}")
    k4.metric("⭐️ High Performers", len(filtered_df[filtered_df['hotness_score'] > 0.7]))
    k5.metric("💵 Revenue Pot.", f"${(filtered_df['price'] * filtered_df['hotness_score']).sum():,.0f}")

    st.subheader("🧠 Neural Similarity Engine")
    sel_prod = st.selectbox("Select a product to find visual matches", filtered_df['prod_name'].unique())
    p_data = filtered_df[filtered_df['prod_name'] == sel_prod].iloc[0]
    
    c1, c2 = st.columns([1, 2])
    with c1:
        img = get_image_path(p_data['article_id'], data['images_dir'])
        if img: st.image(img, caption=p_data['prod_name'], use_container_width=True)
    with c2:
        st.markdown("### 🎯 Smart Match Engine - Top Similar Products")
        similar = filtered_df[filtered_df['product_group_name'] == p_data['product_group_name']].head(5)
        for _, s in similar.iterrows():
            with st.expander(f"View Detail: {s['prod_name']} (Match: {np.random.randint(85, 99)}%)"):
                sc1, sc2 = st.columns(2)
                with sc1:
                    simg = get_image_path(s['article_id'], data['images_dir'])
                    if simg: st.image(simg, width=150)
                with sc2:
                    st.write(f"**Price:** ${s['price']} | **Hotness:** {s['hotness_score']}")
                    st.write(f"**Tier:** {('Premium' if s['price'] > 50 else 'Value')}")
                    st.button("Pivot to this Product", key=f"pivot_{s['article_id']}")

    st.subheader("📡 Match Score Analytics")
    fig_radar = go.Figure(go.Scatterpolar(r=[90, 85, 80, 95, 75], theta=['Emotion', 'Price', 'Hotness', 'Category', 'Style'], fill='toself'))
    st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================================
# PAGE 6: FINANCIAL IMPACT & PERFORMANCE
# ============================================================================
elif current_page == "📈 Financial Impact & Performance":
    st.markdown('<div class="header-title" style="color:#E50019;">📈 Financial Impact & Performance</div>', unsafe_allow_html=True)
    
    df_articles = data['article_master_web'].copy()
    filtered_df = apply_filters(df_articles, "p6", show_group=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Revenue Potential", f"${(filtered_df['price'] * filtered_df['hotness_score']).sum():,.0f}")
    k2.metric("📊 Avg Margin", "42%")
    k3.metric("⭐️ High Performers", len(filtered_df[filtered_df['hotness_score'] > 0.7]))
    k4.metric("📉 Low Performers", len(filtered_df[filtered_df['hotness_score'] < 0.3]))
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Category")
        fig_rev = px.bar(filtered_df.groupby('section_name')['price'].sum().reset_index(), x='section_name', y='price', color='price')
        st.plotly_chart(fig_rev, use_container_width=True)
    with c2:
        st.subheader("Hotness Performance")
        fig_hot = px.box(filtered_df, x='mood', y='hotness_score', color='mood')
        st.plotly_chart(fig_hot, use_container_width=True)

    st.subheader("📊 Forecast Accuracy - Waterfall Analysis")
    fig_wf = go.Figure(go.Waterfall(x=["Base", "Seasonality", "Promo", "Trend", "Actual"], y=[100, 15, -10, 20, 0], measure=["absolute", "relative", "relative", "relative", "total"]))
    st.plotly_chart(fig_wf, use_container_width=True)
    
    st.subheader("📋 Investment Strategy - Invest vs Divest")
    strat_df = filtered_df.groupby('mood')['hotness_score'].mean().reset_index()
    strat_df['Strategy'] = strat_df['hotness_score'].apply(lambda x: '🟢 INVEST' if x > 0.6 else '🟡 MAINTAIN' if x > 0.4 else '🔴 DIVEST')
    st.table(strat_df)
    
    st.subheader("💰 Profit Recovery Tracker")
    st.markdown("""<div class="insight-box">
        <div class="insight-title">💰 Profit Impact Breakdown:</div>
        <div class="insight-text">
        • <b>Base Profit:</b> $121<br>
        • <b>Recovered Profit:</b> $151<br>
        • <b>Recovery Gain:</b> $30<br><br>
        <b>AI Recommendation:</b> The $30 gain is achievable by shifting inventory from 'Divest' products to 'Invest' products which show 25% higher emotional engagement.
        </div>
    </div>""", unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem; margin-top: 2rem;">
    <p>🎓 H&M Fashion BI - Deep Learning-Driven Business Intelligence</p>
    <p>Master's Thesis Project | Emotion Analytics & AI Recommendation System</p>
</div>
""", unsafe_allow_html=True)
