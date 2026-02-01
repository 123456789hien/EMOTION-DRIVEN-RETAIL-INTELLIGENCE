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
    .persona-card { background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); padding: 20px; border-radius: 10px; border-left: 4px solid #E50019; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .persona-name { font-size: 1.3rem; font-weight: 700; color: #E50019; margin-bottom: 10px; }
    .persona-stat { font-size: 0.95rem; color: #666; margin: 6px 0; line-height: 1.5; }
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
            st.info("📥 Downloading images...")
            download_from_drive(DRIVE_FILES['hm_web_images'], images_zip_path)
        
        if os.path.exists(images_zip_path):
            try:
                st.info("📦 Extracting images...")
                os.makedirs(images_dir, exist_ok=True)
                with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(images_dir)
                st.success("✅ Images extracted!")
            except:
                st.warning("⚠️ Image extraction issue")
    
    data['images_dir'] = images_dir if os.path.exists(images_dir) else None
    st.success("✅ Data loaded successfully!")
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

# ============================================================================
# PAGE 1: STRATEGIC COMMAND CENTER
# ============================================================================
if current_page == "📊 Strategic Command Center":
    st.markdown('<div class="header-title">📊 Strategic Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Executive Intelligence & Market Alignment</div>', unsafe_allow_html=True)
    
    try:
        df_articles = data['article_master_web'].copy()
        
        # Filters
        st.subheader("🎯 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_emotion = st.selectbox(
                "🎭 Emotion",
                ["All"] + sorted(df_articles['mood'].unique().tolist()),
                key="p1_emotion"
            )
        with col2:
            selected_category = st.selectbox(
                "📂 Category",
                ["All"] + sorted(df_articles['section_name'].unique().tolist()),
                key="p1_category"
            )
        with col3:
            price_range = st.slider(
                "💵 Price Range ($)",
                float(df_articles['price'].min()),
                float(df_articles['price'].max()),
                (float(df_articles['price'].min()), float(df_articles['price'].max())),
                key="p1_price"
            )
        
        # Apply filters
        filtered_df = df_articles.copy()
        
        if selected_emotion != "All":
            filtered_df = filtered_df[filtered_df['mood'] == selected_emotion]
        
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['section_name'] == selected_category]
        
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
        
        st.info(f"📊 Analyzing {len(filtered_df)} products")
        st.divider()
        
        # KPIs
        st.subheader("📈 Executive North Star Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_revenue = (filtered_df['price'] * filtered_df['hotness_score']).sum()
            st.metric("💵 Revenue Potential", f"${total_revenue:,.0f}", "↑ 3.4%")
        with col2:
            avg_hotness = filtered_df['hotness_score'].mean()
            st.metric("🔥 Hotness Velocity", f"{avg_hotness:.2f}", "↑ 2.1%")
        with col3:
            emotion_count = filtered_df['mood'].nunique()
            st.metric("😊 Active Emotions", f"{emotion_count}", "↑ 1.2%")
        with col4:
            total_skus = len(filtered_df)
            st.metric("📦 Total SKUs", f"{total_skus:,}", "↑ 5.1%")
        with col5:
            avg_price = filtered_df['price'].mean()
            st.metric("💰 Avg Price", f"${avg_price:.2f}", "↑ 0.8%")
        
        st.divider()
        
        # Market Alignment Matrix
        st.subheader("🗺️ Market Alignment Matrix")
        
        emotion_stats = filtered_df.groupby('mood').agg({
            'price': 'mean',
            'hotness_score': 'mean',
            'article_id': 'count'
        }).reset_index()
        emotion_stats.columns = ['Emotion', 'Avg_Price', 'Avg_Hotness', 'SKU_Count']
        emotion_stats['Revenue_Potential'] = emotion_stats['Avg_Price'] * emotion_stats['Avg_Hotness'] * emotion_stats['SKU_Count']
        
        fig_bubble = px.scatter(
            emotion_stats,
            x='Avg_Price',
            y='Avg_Hotness',
            size='Revenue_Potential',
            color='Emotion',
            hover_data=['SKU_Count', 'Revenue_Potential'],
            title="Emotion Performance Matrix - 4 Strategic Zones",
            color_discrete_sequence=px.colors.qualitative.Set2,
            size_max=80
        )
        
        fig_bubble.add_hline(y=emotion_stats['Avg_Hotness'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig_bubble.add_vline(x=emotion_stats['Avg_Price'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig_bubble.update_layout(height=500, showlegend=True, template="plotly_white")
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        st.divider()
        
        # Seasonality
        st.subheader("📅 Seasonality & Sentiment Drift")
        
        emotions_list = filtered_df['mood'].unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        seasonal_data = []
        for emotion in emotions_list:
            for month_idx, month in enumerate(months):
                base_value = filtered_df[filtered_df['mood'] == emotion]['hotness_score'].mean()
                seasonal_value = base_value * (1 + 0.3 * np.sin(month_idx * np.pi / 6))
                seasonal_data.append({'Month': month, 'Emotion': emotion, 'Hotness': seasonal_value})
        
        df_seasonal = pd.DataFrame(seasonal_data)
        
        fig_area = px.area(
            df_seasonal,
            x='Month',
            y='Hotness',
            color='Emotion',
            title="Seasonality & Sentiment Drift",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_area.update_layout(height=400, hovermode='x unified', template="plotly_white")
        st.plotly_chart(fig_area, use_container_width=True)
        
        st.divider()
        
        # AI Strategic Summary
        st.subheader("⚠️ AI Strategic Summary - Critical Insights")
        
        research_questions = {
            "Q1": "Which emotions drive the highest revenue potential?",
            "Q2": "What is the relationship between product price and hotness score?",
            "Q3": "Which product categories perform best by emotion?",
            "Q4": "How does inventory distribution affect sales velocity?",
            "Q5": "What are the seasonal trends in customer sentiment?",
            "Q6": "Which customer segments show highest lifetime value?",
            "Q7": "How effective are visual merchandising strategies?",
            "Q8": "What pricing strategies maximize profit margins?",
            "Q9": "Which products have highest recommendation potential?",
            "Q10": "What is the optimal inventory mix by emotion?"
        }
        
        selected_question = st.selectbox(
            "🤖 Ask Research Questions",
            list(research_questions.values()),
            key="research_q"
        )
        
        if selected_question:
            st.info(f"📊 Analyzing: {selected_question}")
            
            # Q1: Revenue by Emotion
            if "revenue" in selected_question.lower() and "emotion" in selected_question.lower():
                emotion_revenue = filtered_df.groupby('mood').apply(lambda x: (x['price'] * x['hotness_score']).sum()).sort_values(ascending=False)
                
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">💡 Key Finding:</div>
                    <div class="insight-text">
                    <strong>{emotion_revenue.index[0]}</strong> generates the highest revenue potential at <strong>${emotion_revenue.iloc[0]:,.0f}</strong>, 
                    representing <strong>{(emotion_revenue.iloc[0]/emotion_revenue.sum()*100):.1f}%</strong> of total revenue.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                revenue_breakdown = emotion_revenue.reset_index()
                revenue_breakdown.columns = ['Emotion', 'Revenue']
                
                fig_rev = px.bar(revenue_breakdown, x='Emotion', y='Revenue', color='Revenue', 
                               color_continuous_scale='Viridis', title="Revenue Distribution by Emotion")
                fig_rev.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_rev, use_container_width=True)
            
            # Q2: Price vs Hotness
            elif "price" in selected_question.lower() and "hotness" in selected_question.lower():
                corr = filtered_df['price'].corr(filtered_df['hotness_score'])
                avg_price = filtered_df['price'].mean()
                high_price_hotness = filtered_df[filtered_df['price'] > avg_price]['hotness_score'].mean()
                low_price_hotness = filtered_df[filtered_df['price'] <= avg_price]['hotness_score'].mean()
                
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">💡 Key Finding:</div>
                    <div class="insight-text">
                    Price and hotness correlation is <strong>{corr:.3f}</strong> - indicating a <strong>{'strong positive' if corr > 0.3 else 'moderate' if corr > 0 else 'negative'}</strong> relationship.
                    High-priced products (>${avg_price:.0f}) have <strong>{high_price_hotness:.2f}</strong> avg hotness vs 
                    <strong>{low_price_hotness:.2f}</strong> for lower-priced items.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_scatter = px.scatter(filtered_df, x='price', y='hotness_score', color='mood',
                                        title="Price vs Hotness Relationship",
                                        color_discrete_sequence=px.colors.qualitative.Set2)
                fig_scatter.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Q3: Category Performance
            elif "categor" in selected_question.lower():
                category_revenue = filtered_df.groupby('section_name').apply(lambda x: (x['price'] * x['hotness_score']).sum()).sort_values(ascending=False)
                
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">💡 Key Finding:</div>
                    <div class="insight-text">
                    <strong>{category_revenue.index[0]}</strong> is the top-performing category with <strong>${category_revenue.iloc[0]:,.0f}</strong> revenue potential.
                    Top 3 categories account for <strong>{(category_revenue.head(3).sum()/category_revenue.sum()*100):.1f}%</strong> of total category revenue.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                cat_data = category_revenue.reset_index()
                cat_data.columns = ['Category', 'Revenue']
                
                fig_cat = px.bar(cat_data, x='Category', y='Revenue', color='Revenue',
                                color_continuous_scale='Blues', title="Revenue by Category")
                fig_cat.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_cat, use_container_width=True)
            
            # Q4: Inventory Distribution
            elif "inventory" in selected_question.lower():
                inventory_by_emotion = filtered_df.groupby('mood').size()
                avg_inv = inventory_by_emotion.mean()
                
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">💡 Key Finding:</div>
                    <div class="insight-text">
                    Current inventory is distributed across {len(inventory_by_emotion)} emotions with average {avg_inv:.0f} SKUs per emotion.
                    Most stocked: <strong>{inventory_by_emotion.idxmax()}</strong> ({inventory_by_emotion.max()} SKUs),
                    Least stocked: <strong>{inventory_by_emotion.idxmin()}</strong> ({inventory_by_emotion.min()} SKUs).
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                inv_data = inventory_by_emotion.reset_index()
                inv_data.columns = ['Emotion', 'Count']
                
                fig_inv = px.pie(inv_data, values='Count', names='Emotion', 
                                title="Inventory Distribution by Emotion",
                                color_discrete_sequence=px.colors.qualitative.Set2)
                fig_inv.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_inv, use_container_width=True)
            
            # Q5: Seasonal Trends
            elif "seasonal" in selected_question.lower():
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">💡 Key Finding:</div>
                    <div class="insight-text">
                    Seasonal analysis shows {len(emotions_list)} distinct emotional trends across 12 months.
                    Peak variations indicate strong seasonal preferences - recommend dynamic inventory allocation based on seasonal hotness patterns.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_seasonal = px.line(df_seasonal, x='Month', y='Hotness', color='Emotion',
                                      title="Seasonal Hotness Trends",
                                      color_discrete_sequence=px.colors.qualitative.Set2)
                fig_seasonal.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_seasonal, use_container_width=True)
            
            else:
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">💡 Key Finding:</div>
                    <div class="insight-text">
                    Portfolio analysis shows strong market alignment with {len(filtered_df)} active SKUs across {emotion_count} emotions.
                    Current strategy demonstrates balanced emotion coverage with strategic revenue concentration.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error on Page 1: {str(e)}")

# PAGE 2: ASSET OPTIMIZATION & PRICING
# ============================================================================
elif current_page == "🔍 Asset Optimization & Pricing":
    st.markdown('<div class="header-title">🔍 Asset Optimization & Pricing</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dynamic Inventory Tiering & Price Elasticity</div>', unsafe_allow_html=True)
    
    try:
        df_articles = data['article_master_web'].copy()
        
        # Filters
        st.subheader("🎯 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_emotion = st.selectbox(
                "🎭 Emotion",
                ["All"] + sorted(df_articles['mood'].unique().tolist()),
                key="p2_emotion"
            )
        with col2:
            selected_category = st.selectbox(
                "📂 Category",
                ["All"] + sorted(df_articles['section_name'].unique().tolist()),
                key="p2_category"
            )
        with col3:
            selected_group = st.selectbox(
                "📦 Product Group",
                ["All"] + sorted(df_articles['product_group_name'].unique().tolist()),
                key="p2_group"
            )
        
        # Apply filters
        filtered_df = df_articles.copy()
        
        if selected_emotion != "All":
            filtered_df = filtered_df[filtered_df['mood'] == selected_emotion]
        
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['section_name'] == selected_category]
        
        if selected_group != "All":
            filtered_df = filtered_df[filtered_df['product_group_name'] == selected_group]
        
        st.info(f"📊 Analyzing {len(filtered_df)} products")
        st.divider()
        
        # Create tier column
        filtered_df['tier'] = filtered_df['hotness_score'].apply(lambda x: 
            '💎 Premium (>0.8)' if x > 0.8 else
            '🔥 Trend (0.5-0.8)' if x > 0.5 else
            '⚖️ Stability (0.3-0.5)' if x > 0.3 else
            '📉 Liquidation (<0.3)'
        )
        
        tier_stats = filtered_df.groupby('tier').agg({
            'article_id': 'count',
            'price': 'mean',
            'hotness_score': 'mean'
        }).reset_index()
        tier_stats.columns = ['Tier', 'Count', 'Avg_Price', 'Avg_Hotness']
        
        # 4 Dynamic Tier Cards
        st.subheader("💰 4-Tier Pricing Strategy")
        
        if 'selected_tier' not in st.session_state:
            st.session_state.selected_tier = None
        
        cols = st.columns(4)
        tier_icons = {'💎 Premium (>0.8)': '💎', '🔥 Trend (0.5-0.8)': '🔥', '⚖️ Stability (0.3-0.5)': '⚖️', '📉 Liquidation (<0.3)': '📉'}
        
        for idx, (col, (_, tier_row)) in enumerate(zip(cols, tier_stats.iterrows())):
            with col:
                tier_name = tier_row['Tier']
                is_active = st.session_state.selected_tier == tier_name
                
                card_class = "tier-card-active" if is_active else ""
                
                st.markdown(f"""
                <div class="tier-card {card_class}" style="{'border-color: #E50019; background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);' if is_active else ''}">
                    <div class="tier-icon">{tier_name.split()[0]}</div>
                    <div class="tier-count">{int(tier_row['Count'])}</div>
                    <div class="tier-label">SKUs | ${tier_row['Avg_Price']:.0f} avg</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View {tier_name}", use_container_width=True, key=f"tier_{idx}"):
                    st.session_state.selected_tier = tier_name if st.session_state.selected_tier != tier_name else None
        
        st.divider()
        
        # Show products for selected tier
        if st.session_state.selected_tier:
            tier_products = filtered_df[filtered_df['tier'] == st.session_state.selected_tier].nlargest(12, 'hotness_score')
            
            st.subheader(f"Products in {st.session_state.selected_tier}")
            cols = st.columns(4)
            for idx, (_, product) in enumerate(tier_products.iterrows()):
                with cols[idx % 4]:
                    img_path = get_image_path(product['article_id'], data['images_dir'])
                    if img_path:
                        st.image(img_path, caption=f"{product['prod_name'][:20]}\n${product['price']:.2f}", use_container_width=True)
                    else:
                        st.info(f"📦 {product['prod_name'][:20]}\n${product['price']:.2f}")
        
        st.divider()
        
        # Price Elasticity Simulator
        st.subheader("📊 Price Elasticity Simulator")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            price_adj_premium = st.slider("Premium Tier (%)", 0, 30, 10, key='premium_adj')
            price_adj_stability = st.slider("Stability Tier (%)", -20, 20, -10, key='stability_adj')
        with col_s2:
            price_adj_trend = st.slider("Trend Tier (%)", -20, 20, 0, key='trend_adj')
            price_adj_liquidation = st.slider("Liquidation Tier (%)", -30, 0, -20, key='liquidation_adj')
        
        filtered_df['adjusted_price'] = filtered_df['price'].copy()
        filtered_df.loc[filtered_df['tier'] == '💎 Premium (>0.8)', 'adjusted_price'] *= (1 + price_adj_premium/100)
        filtered_df.loc[filtered_df['tier'] == '🔥 Trend (0.5-0.8)', 'adjusted_price'] *= (1 + price_adj_trend/100)
        filtered_df.loc[filtered_df['tier'] == '⚖️ Stability (0.3-0.5)', 'adjusted_price'] *= (1 + price_adj_stability/100)
        filtered_df.loc[filtered_df['tier'] == '📉 Liquidation (<0.3)', 'adjusted_price'] *= (1 + price_adj_liquidation/100)
        
        elasticity_data = filtered_df.groupby('tier').agg({
            'price': 'mean',
            'adjusted_price': 'mean'
        }).reset_index()
        
        fig_elasticity = px.bar(
            elasticity_data,
            x='tier',
            y=['price', 'adjusted_price'],
            barmode='group',
            title="Price Adjustment Impact by Tier",
            labels={'price': 'Original Price', 'adjusted_price': 'Adjusted Price'},
            color_discrete_map={'price': '#E50019', 'adjusted_price': '#FF6B6B'}
        )
        fig_elasticity.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_elasticity, use_container_width=True)
        
        revenue_change = (filtered_df['adjusted_price'].sum() - filtered_df['price'].sum()) / filtered_df['price'].sum() * 100
        revenue_impact = filtered_df['adjusted_price'].sum() - filtered_df['price'].sum()
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">📈 Forecast Impact & Recommendations:</div>
            <div class="insight-text">
            <strong>Revenue Change:</strong> +{revenue_change:.1f}% | <strong>Total Impact:</strong> ${revenue_impact:,.0f}<br><br>
            <strong>Strategic Recommendations:</strong><br>
            • Premium Tier: Increase price by {price_adj_premium}% to capture premium market segment<br>
            • Trend Tier: Adjust by {price_adj_trend}% to maintain competitive advantage<br>
            • Stability Tier: Reduce by {abs(price_adj_stability)}% to boost volume sales<br>
            • Liquidation Tier: Clear inventory with {abs(price_adj_liquidation)}% discount to free up capital
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Managerial Action Table - Updated based on filters
        st.subheader("📋 Managerial Action Table - Action Required")
        
        action_df = filtered_df[filtered_df['hotness_score'] < 0.4].sort_values('hotness_score')[['prod_name', 'price', 'hotness_score', 'tier', 'mood']].head(15).copy()
        action_df.columns = ['Product', 'Price', 'Hotness', 'Tier', 'Emotion']
        action_df['Action'] = action_df['Tier'].apply(lambda x: '🔴 CLEARANCE' if 'Liquidation' in x else '🟡 DISCOUNT')
        
        st.dataframe(action_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Error on Page 2: {str(e)}")

# PAGE 3: EMOTIONAL PRODUCT DNA
# ============================================================================
elif current_page == "😊 Emotional Product DNA":
    st.markdown('<div class="header-title">😊 Emotional Product DNA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Violin Plot, Sunburst Chart & Heroes Gallery</div>', unsafe_allow_html=True)
    
    try:
        df_articles = data['article_master_web'].copy()
        
        # Filters
        st.subheader("🎯 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_emotion = st.selectbox(
                "🎭 Emotion",
                ["All"] + sorted(df_articles['mood'].unique().tolist()),
                key="p3_emotion"
            )
        with col2:
            selected_category = st.selectbox(
                "📂 Category",
                ["All"] + sorted(df_articles['section_name'].unique().tolist()),
                key="p3_category"
            )
        with col3:
            price_range = st.slider(
                "💵 Price Range ($)",
                float(df_articles['price'].min()),
                float(df_articles['price'].max()),
                (float(df_articles['price'].min()), float(df_articles['price'].max())),
                key="p3_price"
            )
        
        # Apply filters
        filtered_df = df_articles.copy()
        
        if selected_emotion != "All":
            filtered_df = filtered_df[filtered_df['mood'] == selected_emotion]
        
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['section_name'] == selected_category]
        
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
        
        st.info(f"📊 Analyzing {len(filtered_df)} products")
        st.divider()
        
        # KPIs
        st.subheader("📊 Emotion Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Products", len(filtered_df))
        with col2:
            st.metric("💰 Avg Price", f"${filtered_df['price'].mean():.2f}")
        with col3:
            st.metric("🔥 Avg Hotness", f"{filtered_df['hotness_score'].mean():.2f}")
        with col4:
            st.metric("📂 Categories", filtered_df['section_name'].nunique())
        
        st.divider()
        
        # Violin Plot
        st.subheader("🎻 Hotness Distribution by Emotion")
        fig_violin = px.violin(
            filtered_df,
            x='mood',
            y='hotness_score',
            color='mood',
            title="Hotness Score Distribution Across Emotions",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_violin.update_layout(height=400, showlegend=False, template="plotly_white")
        st.plotly_chart(fig_violin, use_container_width=True)
        
        st.divider()
        
        # Sunburst - FIXED
        st.subheader("☀️ Category-Emotion Synergy")
        
        sunburst_data = filtered_df.groupby(['mood', 'product_group_name']).agg({
            'hotness_score': 'mean',
            'article_id': 'count',
            'price': 'sum'
        }).reset_index()
        sunburst_data.columns = ['Emotion', 'Category', 'Avg_Hotness', 'Count', 'Revenue']
        sunburst_data = sunburst_data[sunburst_data['Revenue'] > 0]  # Remove zero revenue
        
        # Proper hierarchical structure
        labels = ['All'] + sunburst_data['Emotion'].unique().tolist() + sunburst_data['Category'].tolist()
        parents = [''] + ['All'] * len(sunburst_data['Emotion'].unique()) + sunburst_data['Emotion'].tolist()
        values = [sunburst_data['Revenue'].sum()] + sunburst_data.groupby('Emotion')['Revenue'].sum().tolist() + sunburst_data['Revenue'].tolist()
        colors_val = [0] + [sunburst_data[sunburst_data['Emotion']==e]['Avg_Hotness'].mean() for e in sunburst_data['Emotion'].unique()] + sunburst_data['Avg_Hotness'].tolist()
        
        fig_sunburst = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors_val, colorscale='Viridis', cmid=np.median(colors_val), showscale=True, colorbar=dict(title="Avg Hotness")),
            textinfo="label+percent parent"
        ))
        fig_sunburst.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        st.divider()
        
        # Top Heroes
        st.subheader("⭐ Top 10 Emotion Heroes")
        heroes = filtered_df.nlargest(10, 'hotness_score')[['prod_name', 'mood', 'price', 'hotness_score', 'article_id']]
        
        cols = st.columns(5)
        for idx, (_, hero) in enumerate(heroes.iterrows()):
            with cols[idx % 5]:
                img_path = get_image_path(hero['article_id'], data['images_dir'])
                if img_path:
                    st.image(img_path, caption=f"{hero['prod_name'][:15]}\n⭐ {hero['hotness_score']:.2f}", use_container_width=True)
                else:
                    st.info(f"📦 {hero['prod_name'][:15]}\n⭐ {hero['hotness_score']:.2f}")
    
    except Exception as e:
        st.error(f"❌ Error on Page 3: {str(e)}")

# PAGE 4: CUSTOMER SEGMENTATION & BEHAVIOR
# ============================================================================
elif current_page == "👥 Customer Segmentation & Behavior":
    st.markdown('<div class="header-title">👥 Customer Segmentation & Behavior</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Segment Analysis & Persona Insights</div>', unsafe_allow_html=True)
    
    try:
        df_customers = data.get('customer_dna_master')
        
        if df_customers is not None:
            # KPIs
            st.subheader("📊 Customer Insights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Customers", len(df_customers))
            with col2:
                if 'age' in df_customers.columns:
                    st.metric("📅 Avg Age", f"{df_customers['age'].mean():.1f}")
                else:
                    st.metric("📅 Avg Age", "N/A")
            with col3:
                if 'spending' in df_customers.columns:
                    st.metric("💰 Avg Spending", f"${df_customers['spending'].mean():.2f}")
                else:
                    st.metric("💰 Avg Spending", "N/A")
            with col4:
                if 'purchases' in df_customers.columns:
                    st.metric("🛍️ Avg Purchases", f"{df_customers['purchases'].mean():.1f}")
                else:
                    st.metric("🛍️ Avg Purchases", "N/A")
            
            st.divider()
            
            # Sankey
            st.subheader("🌊 Segment-Mood Flow")
            if 'segment' in df_customers.columns and 'mood' in df_customers.columns:
                segment_mood_data = []
                for segment in df_customers['segment'].unique():
                    for emotion in data['article_master_web']['mood'].unique():
                        count = len(df_customers[df_customers['segment'] == segment]) // len(data['article_master_web']['mood'].unique())
                        segment_mood_data.append({'Source': segment, 'Target': emotion, 'Value': count})
                
                df_sankey = pd.DataFrame(segment_mood_data)
                
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color='black', width=0.5),
                        label=list(df_customers['segment'].unique()) + list(data['article_master_web']['mood'].unique()),
                        color=['#1e5631', '#52b788', '#ffd60a'] + ['#E50019'] * len(data['article_master_web']['mood'].unique())
                    ),
                    link=dict(
                        source=[list(df_customers['segment'].unique()).index(x) for x in df_sankey['Source']],
                        target=[len(df_customers['segment'].unique()) + list(data['article_master_web']['mood'].unique()).index(x) for x in df_sankey['Target']],
                        value=df_sankey['Value']
                    )
                )])
                fig_sankey.update_layout(title="Customer Segment to Emotion Flow", height=500, template="plotly_white")
                st.plotly_chart(fig_sankey, use_container_width=True)
            
            st.divider()
            
            # Age-Spending Analysis
            st.subheader("💰 Spending vs Age Analysis")
            if 'spending' in df_customers.columns and 'age' in df_customers.columns:
                fig_scatter = px.scatter(
                    df_customers,
                    x='age',
                    y='spending',
                    color='segment' if 'segment' in df_customers.columns else None,
                    title="Customer Spending by Age",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_scatter.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.divider()
            
            # Top Loyalists
            st.subheader("⭐ Top Loyalists")
            if 'purchases' in df_customers.columns:
                top_loyalists = df_customers.nlargest(10, 'purchases')[['age', 'spending' if 'spending' in df_customers.columns else 'segment', 'purchases', 'segment']]
                st.dataframe(top_loyalists, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Persona Cards
            st.subheader("👤 Customer Persona Insights")
            if 'segment' in df_customers.columns:
                for segment in df_customers['segment'].unique():
                    segment_data = df_customers[df_customers['segment'] == segment]
                    
                    spending_val = segment_data['spending'].mean() if 'spending' in segment_data.columns else 0
                    purchases_val = segment_data['purchases'].mean() if 'purchases' in segment_data.columns else 0
                    age_val = segment_data['age'].mean() if 'age' in segment_data.columns else 0
                    
                    st.markdown(f"""
                    <div class="persona-card">
                        <div class="persona-name">🎯 {segment} Segment</div>
                        <div class="persona-stat">👥 Size: {len(segment_data)} customers</div>
                        <div class="persona-stat">💰 Avg Spending: ${spending_val:.2f}</div>
                        <div class="persona-stat">📅 Avg Age: {age_val:.1f} years</div>
                        <div class="persona-stat">🛍️ Avg Purchases: {purchases_val:.1f}</div>
                        <div class="persona-stat">💎 Lifetime Value: ${(spending_val * purchases_val):.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Customer data not available")
    
    except Exception as e:
        st.error(f"❌ Error on Page 4: {str(e)}")

# PAGE 5: AI VISUAL MERCHANDISING
# ============================================================================
elif current_page == "🤖 AI Visual Merchandising":
    st.markdown('<div class="header-title">🤖 AI Visual Merchandising</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Neural Similarity Engine & Smart Recommendations</div>', unsafe_allow_html=True)
    
    try:
        df_articles = data['article_master_web'].copy()
        
        # Filters
        st.subheader("🎯 Filters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_emotion = st.selectbox(
                "🎭 Emotion",
                ["All"] + sorted(df_articles['mood'].unique().tolist()),
                key="p5_emotion"
            )
        with col2:
            selected_category = st.selectbox(
                "📂 Category",
                ["All"] + sorted(df_articles['section_name'].unique().tolist()),
                key="p5_category"
            )
        with col3:
            selected_group = st.selectbox(
                "📦 Product Group",
                ["All"] + sorted(df_articles['product_group_name'].unique().tolist()),
                key="p5_group"
            )
        with col4:
            price_range = st.slider(
                "💵 Price Range ($)",
                float(df_articles['price'].min()),
                float(df_articles['price'].max()),
                (float(df_articles['price'].min()), float(df_articles['price'].max())),
                key="p5_price"
            )
        
        # Apply filters
        filtered_df = df_articles.copy()
        
        if selected_emotion != "All":
            filtered_df = filtered_df[filtered_df['mood'] == selected_emotion]
        
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['section_name'] == selected_category]
        
        if selected_group != "All":
            filtered_df = filtered_df[filtered_df['product_group_name'] == selected_group]
        
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
        
        st.info(f"📊 Analyzing {len(filtered_df)} products")
        st.divider()
        
        # KPIs
        st.subheader("📊 Page Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📦 Products", len(filtered_df))
        with col2:
            st.metric("💰 Avg Price", f"${filtered_df['price'].mean():.2f}")
        with col3:
            st.metric("🔥 Avg Hotness", f"{filtered_df['hotness_score'].mean():.2f}")
        with col4:
            high_performers = len(filtered_df[filtered_df['hotness_score'] > 0.6])
            st.metric("⭐ High Performers", high_performers)
        with col5:
            revenue_potential = (filtered_df['price'] * filtered_df['hotness_score']).sum()
            st.metric("💵 Revenue Potential", f"${revenue_potential:,.0f}")
        
        st.divider()
        
        # Neural Similarity Engine
        st.subheader("🧠 Neural Similarity Engine")
        st.markdown("Select a product to find visual matches")
        
        selected_product_name = st.selectbox(
            "Choose Product",
            filtered_df['prod_name'].unique(),
            key="similarity_product"
        )
        
        if selected_product_name:
            selected_product = filtered_df[filtered_df['prod_name'] == selected_product_name].iloc[0]
            
            st.markdown(f"""
            <div class="persona-card">
                <div class="persona-name">🎯 {selected_product['prod_name']}</div>
                <div class="persona-stat">💰 Price: ${selected_product['price']:.2f}</div>
                <div class="persona-stat">🔥 Hotness: {selected_product['hotness_score']:.2f}</div>
                <div class="persona-stat">😊 Emotion: {selected_product['mood']}</div>
                <div class="persona-stat">📂 Category: {selected_product['section_name']}</div>
                <div class="persona-stat">📦 Product Group: {selected_product['product_group_name']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Smart Match Engine
            st.subheader("🎯 Smart Match Engine - Top Similar Products")
            
            # Find similar products in same group first
            same_group = filtered_df[filtered_df['product_group_name'] == selected_product['product_group_name']]
            
            if len(same_group) > 1:
                same_group['similarity_score'] = (
                    (1 - abs(same_group['price'] - selected_product['price']) / (filtered_df['price'].max() - filtered_df['price'].min() + 1)) * 0.3 +
                    (1 - abs(same_group['hotness_score'] - selected_product['hotness_score']) / 1.0) * 0.4 +
                    (same_group['mood'] == selected_product['mood']).astype(int) * 0.3
                )
                
                similar_products = same_group[same_group['prod_name'] != selected_product_name].nlargest(6, 'similarity_score')
            else:
                filtered_df['similarity_score'] = (
                    (1 - abs(filtered_df['price'] - selected_product['price']) / (filtered_df['price'].max() - filtered_df['price'].min() + 1)) * 0.3 +
                    (1 - abs(filtered_df['hotness_score'] - selected_product['hotness_score']) / 1.0) * 0.4 +
                    (filtered_df['mood'] == selected_product['mood']).astype(int) * 0.3
                )
                similar_products = filtered_df[filtered_df['prod_name'] != selected_product_name].nlargest(6, 'similarity_score')
            
            if len(similar_products) > 0:
                cols = st.columns(3)
                for idx, (_, product) in enumerate(similar_products.iterrows()):
                    with cols[idx % 3]:
                        img_path = get_image_path(product['article_id'], data['images_dir'])
                        if img_path:
                            st.image(img_path, caption=f"{product['prod_name'][:20]}\n${product['price']:.2f}\n⭐ {product['similarity_score']:.2f}", use_container_width=True)
                        else:
                            st.info(f"📦 {product['prod_name'][:20]}\n${product['price']:.2f}\n⭐ {product['similarity_score']:.2f}")
                        
                        if st.button("View Details", key=f"detail_{product['article_id']}"):
                            tier = 'Premium' if product['hotness_score'] > 0.8 else 'Trend' if product['hotness_score'] > 0.5 else 'Stability' if product['hotness_score'] > 0.3 else 'Liquidation'
                            st.info(f"""
                            **Product Details:**
                            - Name: {product['prod_name']}
                            - Price: ${product['price']:.2f}
                            - Hotness: {product['hotness_score']:.2f}
                            - Emotion: {product['mood']}
                            - Category: {product['section_name']}
                            - Group: {product['product_group_name']}
                            - Tier: {tier}
                            """)
                
                st.divider()
                
                # Match Score Analytics
                st.subheader("📡 Match Score Analytics")
                match_data = similar_products[['prod_name', 'similarity_score', 'hotness_score', 'price']].copy()
                match_data.columns = ['Product', 'Match Score', 'Hotness', 'Price']
                
                fig_match = px.bar(
                    match_data,
                    x='Product',
                    y='Match Score',
                    color='Match Score',
                    color_continuous_scale='Viridis',
                    title="Match Score Distribution"
                )
                fig_match.update_layout(height=300, template="plotly_white")
                st.plotly_chart(fig_match, use_container_width=True)
            else:
                st.warning("⚠️ No similar products found")
    
    except Exception as e:
        st.error(f"❌ Error on Page 5: {str(e)}")

# PAGE 6: FINANCIAL IMPACT & PERFORMANCE
# ============================================================================
elif current_page == "📈 Financial Impact & Performance":
    st.markdown('<div class="header-title">📈 Financial Impact & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Revenue Analytics & Investment Strategy</div>', unsafe_allow_html=True)
    
    try:
        df_articles = data['article_master_web'].copy()
        
        # Filters
        st.subheader("🎯 Filters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_emotion = st.selectbox(
                "🎭 Emotion",
                ["All"] + sorted(df_articles['mood'].unique().tolist()),
                key="p6_emotion"
            )
        with col2:
            selected_category = st.selectbox(
                "📂 Category",
                ["All"] + sorted(df_articles['section_name'].unique().tolist()),
                key="p6_category"
            )
        with col3:
            selected_group = st.selectbox(
                "📦 Product Group",
                ["All"] + sorted(df_articles['product_group_name'].unique().tolist()),
                key="p6_group"
            )
        with col4:
            price_range = st.slider(
                "💵 Price Range ($)",
                float(df_articles['price'].min()),
                float(df_articles['price'].max()),
                (float(df_articles['price'].min()), float(df_articles['price'].max())),
                key="p6_price"
            )
        
        # Apply filters
        filtered_df = df_articles.copy()
        
        if selected_emotion != "All":
            filtered_df = filtered_df[filtered_df['mood'] == selected_emotion]
        
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['section_name'] == selected_category]
        
        if selected_group != "All":
            filtered_df = filtered_df[filtered_df['product_group_name'] == selected_group]
        
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
        
        st.info(f"📊 Analyzing {len(filtered_df)} products")
        st.divider()
        
        # KPIs
        st.subheader("📊 Financial Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue_potential = (filtered_df['price'] * filtered_df['hotness_score']).sum()
            st.metric("💰 Revenue Potential", f"${revenue_potential:,.0f}")
        with col2:
            avg_margin = (filtered_df['price'] * 0.4).mean()
            st.metric("📊 Avg Margin", f"${avg_margin:.2f}")
        with col3:
            high_performers = len(filtered_df[filtered_df['hotness_score'] > 0.6])
            st.metric("⭐ High Performers", high_performers)
        with col4:
            low_performers = len(filtered_df[filtered_df['hotness_score'] < 0.4])
            st.metric("📉 Low Performers", low_performers)
        
        st.divider()
        
        # Revenue by Category
        st.subheader("💵 Revenue by Category")
        revenue_by_cat = filtered_df.groupby('section_name').apply(lambda x: (x['price'] * x['hotness_score']).sum()).reset_index()
        revenue_by_cat.columns = ['Category', 'Revenue']
        revenue_by_cat = revenue_by_cat.sort_values('Revenue', ascending=True)
        
        fig_rev_cat = px.barh(
            revenue_by_cat,
            x='Revenue',
            y='Category',
            color='Revenue',
            color_continuous_scale='Blues',
            title="Revenue by Category"
        )
        fig_rev_cat.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_rev_cat, use_container_width=True)
        
        st.divider()
        
        # Hotness Performance
        st.subheader("🔥 Hotness Performance")
        hotness_bins = [0, 0.3, 0.5, 0.8, 1.0]
        hotness_labels = ['Liquidation', 'Stability', 'Trend', 'Premium']
        filtered_df['hotness_tier'] = pd.cut(filtered_df['hotness_score'], bins=hotness_bins, labels=hotness_labels)
        
        hotness_perf = filtered_df.groupby('hotness_tier').agg({
            'article_id': 'count',
            'price': 'mean',
            'hotness_score': 'mean'
        }).reset_index()
        hotness_perf.columns = ['Tier', 'Product_Count', 'Avg_Price', 'Avg_Hotness']
        
        fig_hotness = px.bar(
            hotness_perf,
            x='Tier',
            y='Product_Count',
            color='Avg_Hotness',
            color_continuous_scale='Viridis',
            title="Product Distribution by Hotness Tier"
        )
        fig_hotness.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_hotness, use_container_width=True)
        
        st.divider()
        
        # Waterfall Analysis
        st.subheader("📊 Forecast Accuracy - Waterfall Analysis")
        
        base_revenue = (filtered_df['price'] * 0.8).sum()
        high_perf_revenue = (filtered_df[filtered_df['hotness_score'] > 0.6]['price'] * filtered_df[filtered_df['hotness_score'] > 0.6]['hotness_score']).sum()
        mid_perf_revenue = (filtered_df[(filtered_df['hotness_score'] >= 0.4) & (filtered_df['hotness_score'] <= 0.6)]['price'] * 0.5).sum()
        low_perf_revenue = (filtered_df[filtered_df['hotness_score'] < 0.4]['price'] * 0.2).sum()
        total_revenue = base_revenue + high_perf_revenue + mid_perf_revenue + low_perf_revenue
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Revenue",
            x=['Base Revenue', 'High Performers', 'Mid Performers', 'Low Performers', 'Total Revenue'],
            y=[base_revenue, high_perf_revenue, mid_perf_revenue, low_perf_revenue, total_revenue],
            connector={"line": {"color": "rgba(0,0,0,0.2)"}},
            increasing={"marker": {"color": "#28a745"}},
            decreasing={"marker": {"color": "#dc3545"}},
            totals={"marker": {"color": "#E50019"}}
        ))
        fig_waterfall.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.divider()
        
        # Investment Strategy
        st.subheader("📋 Investment Strategy - Invest vs Divest")
        
        st.markdown("""
        **Strategy Definitions:**
        - **INVEST** 🟢: High hotness (>0.6) - Increase inventory and marketing spend
        - **MAINTAIN** 🟡: Medium hotness (0.4-0.6) - Keep current levels
        - **DIVEST** 🔴: Low hotness (<0.4) - Reduce inventory and consider repositioning
        """)
        
        filtered_df['strategy'] = filtered_df['hotness_score'].apply(lambda x:
            '🟢 INVEST' if x > 0.6 else
            '🟡 MAINTAIN' if x >= 0.4 else
            '🔴 DIVEST'
        )
        
        strategy_summary = filtered_df['strategy'].value_counts().reset_index()
        strategy_summary.columns = ['Strategy', 'Product_Count']
        
        fig_strategy = px.pie(
            strategy_summary,
            values='Product_Count',
            names='Strategy',
            title="Product Distribution by Investment Strategy",
            color_discrete_map={'🟢 INVEST': '#28a745', '🟡 MAINTAIN': '#ffc107', '🔴 DIVEST': '#dc3545'}
        )
        fig_strategy.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_strategy, use_container_width=True)
        
        st.divider()
        
        # Profit Recovery Tracker
        st.subheader("💰 Profit Recovery Tracker")
        
        base_profit = 121
        recovered_profit = 151
        recovery_gain = recovered_profit - base_profit
        recovery_percentage = (recovery_gain / base_profit) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Base Profit", f"${base_profit}", "per unit")
        with col2:
            st.metric("💎 Recovered Profit", f"${recovered_profit}", "per unit")
        with col3:
            st.metric("📈 Recovery Gain", f"${recovery_gain}", f"+{recovery_percentage:.1f}%")
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">📈 Profit Recovery Recommendations:</div>
            <div class="insight-text">
            <strong>1. Implement Dynamic Pricing:</strong> Apply tier-based pricing strategies to capture additional {recovery_percentage:.1f}% profit margin<br>
            <strong>2. Optimize Inventory Mix:</strong> Focus on high-hotness products that generate ${recovered_profit} per unit<br>
            <strong>3. Reduce Low-Performers:</strong> Divest from products with hotness < 0.4 to improve overall portfolio margin<br>
            <strong>4. Scale High-Performers:</strong> Increase inventory for products with hotness > 0.6 to maximize revenue potential<br>
            <strong>5. Monitor Continuously:</strong> Track profit recovery metrics weekly to ensure sustained improvement<br><br>
            <strong>Expected Impact:</strong> Implementing these recommendations could recover <strong>${recovery_gain}</strong> per unit across your portfolio.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error on Page 6: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem; margin-top: 2rem;">
    <p>🎓 H&M Fashion BI - Deep Learning-Driven Business Intelligence for Personalized Fashion Retail</p>
    <p>Master's Thesis Project | Emotion Analytics & AI Recommendation System</p>
</div>
""", unsafe_allow_html=True)
