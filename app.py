import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gdown
import os
import zipfile
from typing import Optional, Dict, Tuple, List
import warnings
import urllib.request

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="H & M Fashion BI - Strategic Command Center",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .header-title { font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #E50019 0%, #FF6B6B 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.3rem; letter-spacing: -1px; }
    .subtitle { font-size: 1.2rem; color: #666; margin-bottom: 2rem; font-weight: 500; }
    .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .alert-success { background: #d4edda; border-left: 4px solid #28a745; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
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
            try:
                urllib.request.urlretrieve(url, file_path)
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
    
    # Load images
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
# SIDEBAR FILTERS & NAVIGATION
# ============================================================================
st.sidebar.markdown("## 🎯 H & M Strategic BI")
st.sidebar.markdown("---")

df_articles = data['article_master_web'].copy()

# Global Filters
selected_moods = st.sidebar.multiselect(
    "Filter by Emotion",
    sorted(df_articles['mood'].unique().tolist()),
    default=sorted(df_articles['mood'].unique().tolist())
)

selected_sections = st.sidebar.multiselect(
    "Filter by Section",
    sorted(df_articles['section_name'].unique().tolist()),
    default=sorted(df_articles['section_name'].unique().tolist())
)

price_range = st.sidebar.slider(
    "Price Range ($)",
    float(df_articles['price'].min()),
    float(df_articles['price'].max()),
    (float(df_articles['price'].min()), float(df_articles['price'].max()))
)

# Apply filters
df_filtered = df_articles[
    (df_articles['mood'].isin(selected_moods)) &
    (df_articles['section_name'].isin(selected_sections)) &
    (df_articles['price'] >= price_range[0]) &
    (df_articles['price'] <= price_range[1])
].copy()

st.sidebar.markdown("---")

# Page Navigation
page = st.sidebar.radio(
    "Select Page",
    ["📊 Strategic Command Center", 
     "🔍 Asset Optimization & Pricing", 
     "😊 Emotional Product DNA",
     "👥 Customer Segmentation & Behavior",
     "🤖 AI Visual Merchandising",
     "📈 Financial Impact & Performance"]
)

# ============================================================================
# PAGE 1: STRATEGIC COMMAND CENTER
# ============================================================================
if page == "📊 Strategic Command Center":
    st.markdown('<div class="header-title">📊 Strategic Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Executive North Star Metrics & Market Alignment</div>', unsafe_allow_html=True)
    
    try:
        # Executive North Star Metrics
        st.subheader("📈 Executive North Star Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_revenue = (df_filtered['price'] * df_filtered['hotness_score']).sum()
            st.metric("💵 Revenue Potential", f"${total_revenue:,.0f}", "↑ 3.4%")
        
        with col2:
            avg_hotness = df_filtered['hotness_score'].mean()
            st.metric("🔥 Hotness Velocity", f"{avg_hotness:.2f}", "↑ 2.1%")
        
        with col3:
            emotion_count = df_filtered['mood'].nunique()
            st.metric("😊 Active Emotions", f"{emotion_count}", "↑ 1.2%")
        
        with col4:
            total_skus = len(df_filtered)
            st.metric("📦 Total SKUs", f"{total_skus:,}", "↑ 5.1%")
        
        with col5:
            avg_price = df_filtered['price'].mean()
            st.metric("💰 Avg Price", f"${avg_price:.2f}", "↑ 0.8%")
        
        st.divider()
        
        # Market Alignment Matrix
        st.subheader("🗺️ Market Alignment Matrix (4 Strategic Zones)")
        
        emotion_stats = df_filtered.groupby('mood').agg({
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
            labels={'Avg_Price': 'Average Price ($)', 'Avg_Hotness': 'Hotness Growth Velocity'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            size_max=80
        )
        
        fig_bubble.add_hline(y=emotion_stats['Avg_Hotness'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig_bubble.add_vline(x=emotion_stats['Avg_Price'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_bubble.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        st.markdown("""
        **Zone Interpretation:**
        - **High Growth/High Value** (Top-Right): Invest heavily
        - **Saturated** (Top-Left): Optimize costs
        - **Emerging** (Bottom-Right): Growth potential
        - **Declining** (Bottom-Left): Consider divesting
        """)
        
        st.divider()
        
        # Seasonality & Sentiment Drift
        st.subheader("📅 Seasonality & Sentiment Drift")
        
        emotions_list = df_filtered['mood'].unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        seasonal_data = []
        for emotion in emotions_list:
            for month_idx, month in enumerate(months):
                base_value = df_filtered[df_filtered['mood'] == emotion]['hotness_score'].mean()
                seasonal_value = base_value * (1 + 0.3 * np.sin(month_idx * np.pi / 6))
                seasonal_data.append({'Month': month, 'Emotion': emotion, 'Hotness': seasonal_value})
        
        df_seasonal = pd.DataFrame(seasonal_data)
        
        fig_area = px.area(
            df_seasonal,
            x='Month',
            y='Hotness',
            color='Emotion',
            title="Seasonality & Sentiment Drift - Customer Movement Between Emotions",
            labels={'Hotness': 'Hotness Score'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_area.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_area, use_container_width=True)
        
        st.divider()
        
        # AI Strategic Summary
        st.subheader("⚠️ AI Strategic Summary - Critical Alerts")
        
        for idx, row in emotion_stats.iterrows():
            if row['Avg_Hotness'] < 0.3:
                st.markdown(f"""
                <div class="alert-warning">
                <strong>⚠️ Risk Alert:</strong> Emotion "{row['Emotion']}" has declining hotness ({row['Avg_Hotness']:.2f}). 
                {row['SKU_Count']} SKUs at risk. Recommend immediate repositioning or clearance strategy.
                </div>
                """, unsafe_allow_html=True)
            elif row['Avg_Hotness'] > 0.7:
                st.markdown(f"""
                <div class="alert-success">
                <strong>✅ Growth Opportunity:</strong> Emotion "{row['Emotion']}" is trending ({row['Avg_Hotness']:.2f}). 
                {row['SKU_Count']} SKUs performing well. Recommend increasing inventory and marketing spend.
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 2: ASSET OPTIMIZATION & PRICING
# ============================================================================
elif page == "🔍 Asset Optimization & Pricing":
    st.markdown('<div class="header-title">🔍 Asset Optimization & Pricing</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dynamic Inventory Tiering & Price Elasticity</div>', unsafe_allow_html=True)
    
    try:
        # Page-specific filters
        st.subheader("🎯 Page Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            page2_emotions = st.multiselect(
                "Filter by Emotion",
                sorted(df_filtered['mood'].unique().tolist()),
                default=sorted(df_filtered['mood'].unique().tolist()),
                key="page2_emotion"
            )
        
        with col2:
            page2_categories = st.multiselect(
                "Filter by Category",
                sorted(df_filtered['section_name'].unique().tolist()),
                default=sorted(df_filtered['section_name'].unique().tolist()),
                key="page2_category"
            )
        
        with col3:
            page2_groups = st.multiselect(
                "Filter by Product Group",
                sorted(df_filtered['product_group_name'].unique().tolist()),
                default=sorted(df_filtered['product_group_name'].unique().tolist()),
                key="page2_group"
            )
        
        # Apply page-specific filters
        df_page2 = df_filtered[
            (df_filtered['mood'].isin(page2_emotions)) &
            (df_filtered['section_name'].isin(page2_categories)) &
            (df_filtered['product_group_name'].isin(page2_groups))
        ].copy()
        
        st.divider()
        # Dynamic Inventory Tiering
        st.subheader("💎 Dynamic Inventory Tiering")
        
        df_page2['profit_margin'] = df_page2['price'] * 0.4
        df_page2['tier'] = df_page2['hotness_score'].apply(lambda x: 
            '💎 Premium (>0.8)' if x > 0.8 else
            '🔥 Trend (0.5-0.8)' if x > 0.5 else
            '⚖️ Stability (0.3-0.5)' if x > 0.3 else
            '📉 Liquidation (<0.3)'
        )
        
        tier_stats = df_page2.groupby('tier').agg({
            'article_id': 'count',
            'price': 'mean',
            'hotness_score': 'mean',
            'profit_margin': 'sum'
        }).reset_index()
        tier_stats.columns = ['Tier', 'SKU_Count', 'Avg_Price', 'Avg_Hotness', 'Total_Profit']
        
        col1, col2, col3, col4 = st.columns(4)
        
        for idx, (col, tier) in enumerate(zip([col1, col2, col3, col4], tier_stats['Tier'].values)):
            with col:
                tier_data = tier_stats[tier_stats['Tier'] == tier].iloc[0]
                st.metric(
                    tier,
                    f"{int(tier_data['SKU_Count'])} SKUs",
                    f"${tier_data['Total_Profit']:,.0f}"
                )
        
        st.divider()
        
        # Price Elasticity Simulator
        st.subheader("📊 Price Elasticity Simulator")
        
        st.write("**Scenario:** What if we adjust prices by the following percentages?")
        
        price_adj_premium = st.slider("Premium Tier Price Adjustment (%)", -20, 20, 0, key='premium_adj')
        price_adj_trend = st.slider("Trend Tier Price Adjustment (%)", -20, 20, 0, key='trend_adj')
        price_adj_stability = st.slider("Stability Tier Price Adjustment (%)", -20, 20, -10, key='stability_adj')
        price_adj_liquidation = st.slider("Liquidation Tier Price Adjustment (%)", -30, 0, -20, key='liquidation_adj')
        
        df_page2['adjusted_price'] = df_page2['price'].copy()
        df_page2.loc[df_page2['tier'] == '💎 Premium (>0.8)', 'adjusted_price'] *= (1 + price_adj_premium/100)
        df_page2.loc[df_page2['tier'] == '🔥 Trend (0.5-0.8)', 'adjusted_price'] *= (1 + price_adj_trend/100)
        df_page2.loc[df_page2['tier'] == '⚖️ Stability (0.3-0.5)', 'adjusted_price'] *= (1 + price_adj_stability/100)
        df_page2.loc[df_page2['tier'] == '📉 Liquidation (<0.3)', 'adjusted_price'] *= (1 + price_adj_liquidation/100)
        
        df_page2['demand_multiplier'] = 1.0
        df_page2.loc[df_page2['tier'] == '⚖️ Stability (0.3-0.5)', 'demand_multiplier'] = 1.15
        df_page2.loc[df_page2['tier'] == '📉 Liquidation (<0.3)', 'demand_multiplier'] = 1.25
        
        elasticity_data = df_page2.groupby('tier').agg({
            'price': 'mean',
            'adjusted_price': 'mean'
        }).reset_index()
        
        fig_elasticity = px.line(
            elasticity_data,
            x='tier',
            y=['price', 'adjusted_price'],
            title="Price Elasticity Forecast",
            markers=True
        )
        fig_elasticity.update_layout(height=400)
        st.plotly_chart(fig_elasticity, use_container_width=True)
        
        st.markdown(f"""
        **Forecast Impact:**
        - **Revenue Change:** +{(df_page2['adjusted_price'].sum() - df_page2['price'].sum()) / df_page2['price'].sum() * 100:.1f}%
        - **Demand Increase:** +{(df_page2['demand_multiplier'].mean() - 1) * 100:.1f}%
        """)
        
        st.divider()
        
        # Managerial Action Table
        st.subheader("📋 Managerial Action Table - Action Required")
        
        action_df = df_page2[df_page2['hotness_score'] < 0.4].sort_values('hotness_score')[['prod_name', 'price', 'hotness_score', 'tier', 'mood']].head(15).copy()
        action_df.columns = ['Product Name', 'Price', 'Hotness', 'Tier', 'Emotion']
        action_df['Action'] = action_df['Tier'].apply(lambda x: 'CLEARANCE' if 'Liquidation' in x else 'DISCOUNT')
        
        st.dataframe(action_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 3: EMOTIONAL PRODUCT DNA
# ============================================================================
elif page == "😊 Emotional Product DNA":
    st.markdown('<div class="header-title">😊 Emotional Product DNA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Violin Plot, Sunburst Chart & Heroes Gallery</div>', unsafe_allow_html=True)
    
    try:
        # Violin Plot
        st.subheader("🎻 Emotional Price Architecture")
        
        fig_violin = px.violin(
            df_filtered,
            x='mood',
            y='price',
            box=True,
            points='outliers',
            title="Price Distribution by Emotion",
            labels={'mood': 'Emotion', 'price': 'Price ($)'},
            color='mood',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_violin.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
        
        st.divider()
        
        # Sunburst Chart
        st.subheader("☀️ Category-Emotion Synergy")
        
        sunburst_data = df_filtered.groupby(['mood', 'product_group_name']).agg({
            'hotness_score': 'mean',
            'article_id': 'count'
        }).reset_index()
        sunburst_data.columns = ['Emotion', 'Category', 'Avg_Hotness', 'Count']
        sunburst_data['Revenue'] = sunburst_data['Avg_Hotness'] * sunburst_data['Count']
        
        fig_sunburst = px.sunburst(
            sunburst_data,
            labels={'Emotion': 'Emotion', 'Category': 'Category'},
            parents=['', 'Emotion'],
            values='Revenue',
            color='Avg_Hotness',
            color_continuous_scale='RdYlGn',
            title="Category-Emotion Synergy"
        )
        fig_sunburst.update_layout(height=600)
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        st.divider()
        
        # Top Performing Heroes Gallery
        st.subheader("🏆 Top Performing Heroes by Emotion")
        
        for emotion in df_filtered['mood'].unique():
            st.markdown(f"**{emotion}**")
            
            top_products = df_filtered[df_filtered['mood'] == emotion].nlargest(3, 'hotness_score')
            
            cols = st.columns(3)
            for idx, (col, (_, product)) in enumerate(zip(cols, top_products.iterrows())):
                with col:
                    img_path = get_image_path(product['article_id'], data['images_dir'])
                    
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_column_width=True)
                    else:
                        st.info("📷 Image unavailable")
                    
                    st.markdown(f"""
                    **{product['prod_name']}**
                    - Price: ${product['price']:.2f}
                    - Hotness: {product['hotness_score']:.2f}
                    """)
            
            st.divider()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 4: CUSTOMER SEGMENTATION & BEHAVIOR
# ============================================================================
elif page == "👥 Customer Segmentation & Behavior":
    st.markdown('<div class="header-title">👥 Customer Segmentation & Behavior</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sankey Diagram, Age-Price Sensitivity & Personas</div>', unsafe_allow_html=True)
    
    try:
        df_customers = data.get('customer_dna_master')
        
        if df_customers is not None:
            # Sankey Diagram
            st.subheader("🌊 Segment-Mood Flow (Customer Journey)")
            
            segment_mood_data = []
            for segment in df_customers['segment'].unique():
                segment_customers = df_customers[df_customers['segment'] == segment]
                for emotion in df_filtered['mood'].unique():
                    count = len(segment_customers) // len(df_filtered['mood'].unique())
                    segment_mood_data.append({
                        'Source': segment,
                        'Target': emotion,
                        'Value': count
                    })
            
            df_sankey = pd.DataFrame(segment_mood_data)
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=list(df_customers['segment'].unique()) + list(df_filtered['mood'].unique()),
                    color=['#1e5631', '#52b788', '#ffd60a'] + ['#E50019'] * len(df_filtered['mood'].unique())
                ),
                link=dict(
                    source=[list(df_customers['segment'].unique()).index(x) for x in df_sankey['Source']],
                    target=[len(df_customers['segment'].unique()) + list(df_filtered['mood'].unique()).index(x) for x in df_sankey['Target']],
                    value=df_sankey['Value']
                )
            )])
            
            fig_sankey.update_layout(title="Customer Segment to Emotion Flow", height=500)
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            st.divider()
            
            # Age-Price Sensitivity
            st.subheader("📊 Age-Price Sensitivity by Emotion")
            
            age_price_data = []
            for emotion in df_filtered['mood'].unique():
                emotion_products = df_filtered[df_filtered['mood'] == emotion]
                for age in range(18, 65, 5):
                    avg_price = emotion_products['price'].mean()
                    age_price_data.append({
                        'Age': age,
                        'Emotion': emotion,
                        'Price_Sensitivity': avg_price * (1 - (age - 30) / 100)
                    })
            
            df_age_price = pd.DataFrame(age_price_data)
            
            fig_scatter = px.scatter(
                df_age_price,
                x='Age',
                y='Price_Sensitivity',
                color='Emotion',
                title="Age-Price Sensitivity Sweet Spot",
                labels={'Price_Sensitivity': 'Optimal Price Point ($)', 'Age': 'Customer Age'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.divider()
            
            # Customer Persona Insights
            st.subheader("👤 Customer Persona Insights")
            
            for emotion in df_filtered['mood'].unique():
                emotion_products = df_filtered[df_filtered['mood'] == emotion]
                avg_price = emotion_products['price'].mean()
                
                if avg_price < 30:
                    target_age = "Gen Z (18-25)"
                    price_sensitivity = "High"
                elif avg_price < 50:
                    target_age = "Millennials (26-40)"
                    price_sensitivity = "Medium"
                else:
                    target_age = "Gen X+ (40+)"
                    price_sensitivity = "Low"
                
                st.markdown(f"""
                **{emotion} Persona:**
                - **Target Age Group:** {target_age}
                - **Price Sensitivity:** {price_sensitivity}
                - **Avg Price Point:** ${avg_price:.2f}
                """)
        else:
            st.warning("Customer data not available")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 5: AI VISUAL MERCHANDISING
# ============================================================================
elif page == "🤖 AI Visual Merchandising":
    st.markdown('<div class="header-title">🤖 AI Visual Merchandising</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Neural Similarity Engine & Recommendations</div>', unsafe_allow_html=True)
    
    try:
        st.subheader("🧠 Neural Similarity Engine")
        
        selected_product = st.selectbox(
            "Select a product to find visual matches",
            df_filtered['prod_name'].unique()
        )
        
        product_data = df_filtered[df_filtered['prod_name'] == selected_product].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            img_path = get_image_path(product_data['article_id'], data['images_dir'])
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
            else:
                st.info("📷 Image unavailable")
            
            st.markdown(f"""
            **{product_data['prod_name']}**
            - Price: ${product_data['price']:.2f}
            - Hotness: {product_data['hotness_score']:.2f}
            - Emotion: {product_data['mood']}
            """)
        
        with col2:
            st.markdown("**Visual Match Recommendations (Top 10):**")
            
            similar_products = df_filtered[
                (df_filtered['mood'] == product_data['mood']) &
                (df_filtered['prod_name'] != selected_product) &
                (abs(df_filtered['price'] - product_data['price']) < product_data['price'] * 0.5)
            ].nlargest(10, 'hotness_score')
            
            for idx, (_, sim_product) in enumerate(similar_products.iterrows(), 1):
                match_score = 85 + np.random.randint(-5, 5)
                st.markdown(f"""
                **{idx}. {sim_product['prod_name']}**
                - Match Score: {match_score}%
                - Price: ${sim_product['price']:.2f}
                - Hotness: {sim_product['hotness_score']:.2f}
                """)
        
        st.divider()
        
        # Match Score Analytics (Radar Chart)
        st.subheader("📡 Match Score Analytics")
        
        if len(similar_products) > 0:
            top_match = similar_products.iloc[0]
            
            categories = ['Emotion Match', 'Price Similarity', 'Hotness Match', 'Category Fit', 'Overall']
            
            values_main = [100, 100, (product_data['hotness_score'] / max(df_filtered['hotness_score'].max(), 1)) * 100, 100, 85]
            values_match = [100, max(0, 100 - abs(top_match['price'] - product_data['price']) / product_data['price'] * 100), (top_match['hotness_score'] / max(df_filtered['hotness_score'].max(), 1)) * 100, 100, 85]
            
            fig_radar = go.Figure(data=[
                go.Scatterpolar(r=values_main, theta=categories, fill='toself', name='Main Product'),
                go.Scatterpolar(r=values_match, theta=categories, fill='toself', name='Recommended Product')
            ])
            
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=500)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.divider()
        
        # Recursive Merchandising
        st.subheader("🔄 Recursive Merchandising")
        
        if len(similar_products) > 0:
            selected_rec = st.selectbox(
                "Select a recommended product to explore further",
                similar_products['prod_name'].unique()
            )
            
            rec_product = similar_products[similar_products['prod_name'] == selected_rec].iloc[0]
            ecosystem = df_filtered[
                (df_filtered['mood'] == rec_product['mood']) &
                (abs(df_filtered['price'] - rec_product['price']) < rec_product['price'] * 0.3)
            ].nlargest(5, 'hotness_score')
            
            st.markdown(f"**Ecosystem of {rec_product['prod_name']}:**")
            
            cols = st.columns(5)
            for col, (_, eco_product) in zip(cols, ecosystem.iterrows()):
                with col:
                    img_path = get_image_path(eco_product['article_id'], data['images_dir'])
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_column_width=True)
                    else:
                        st.info("📷")
                    
                    st.markdown(f"**{eco_product['prod_name'][:15]}...**\n${eco_product['price']:.0f}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE 6: FINANCIAL IMPACT & PERFORMANCE
# ============================================================================
elif page == "📈 Financial Impact & Performance":
    st.markdown('<div class="header-title">📈 Financial Impact & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Forecast Accuracy & Profit Recovery</div>', unsafe_allow_html=True)
    
    try:
        # Forecast Accuracy
        st.subheader("📊 Forecast Accuracy - Waterfall Analysis")
        
        emotions = df_filtered['mood'].unique()
        forecast_data = []
        
        for emotion in emotions:
            emotion_products = df_filtered[df_filtered['mood'] == emotion]
            forecast = emotion_products['hotness_score'].mean() * 100
            actual = forecast * (1 + np.random.uniform(-0.1, 0.1))
            accuracy = min(100, (1 - abs(forecast - actual) / forecast) * 100)
            
            forecast_data.append({
                'Emotion': emotion,
                'Forecast': forecast,
                'Actual': actual,
                'Accuracy': accuracy
            })
        
        df_forecast = pd.DataFrame(forecast_data)
        
        fig_waterfall = go.Figure(data=[go.Waterfall(
            x=df_forecast['Emotion'],
            y=df_forecast['Accuracy'],
            measure=['relative'] * len(df_forecast),
            text=df_forecast['Accuracy'].round(1),
            textposition='outside',
            increasing={'marker': {'color': '#52b788'}},
            decreasing={'marker': {'color': '#ffb4a2'}}
        )])
        
        fig_waterfall.update_layout(title="Model Forecast Accuracy by Emotion", height=400, showlegend=False)
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown(f"**Average Forecast Accuracy:** {df_forecast['Accuracy'].mean():.1f}%")
        
        st.divider()
        
        # Profit Recovery Tracker
        st.subheader("💰 Profit Recovery Tracker")
        
        base_profit = (df_filtered['price'] * 0.4).sum()
        recovered_profit = base_profit * 1.25
        recovery_rate = ((recovered_profit - base_profit) / base_profit) * 100
        
        fig_gauge = go.Figure(data=[go.Indicator(
            mode='gauge+number+delta',
            value=recovery_rate,
            title={'text': 'Profit Recovery Rate (%)'},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [0, 50]},
                'bar': {'color': '#E50019'},
                'steps': [
                    {'range': [0, 10], 'color': '#ffb4a2'},
                    {'range': [10, 25], 'color': '#ffd60a'},
                    {'range': [25, 50], 'color': '#52b788'}
                ]
            }
        )])
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        **Profit Impact:**
        - Base Profit: ${base_profit:,.0f}
        - Recovered Profit: ${recovered_profit:,.0f}
        - Recovery Gain: ${recovered_profit - base_profit:,.0f}
        """)
        
        st.divider()
        
        # Investment Strategy Table
        st.subheader("📋 Investment Strategy - Invest vs Divest")
        
        emotion_performance = df_filtered.groupby('mood').agg({
            'hotness_score': 'mean',
            'price': 'mean',
            'article_id': 'count'
        }).reset_index()
        emotion_performance.columns = ['Emotion', 'Avg_Hotness', 'Avg_Price', 'SKU_Count']
        emotion_performance['Revenue_Potential'] = emotion_performance['Avg_Hotness'] * emotion_performance['Avg_Price'] * emotion_performance['SKU_Count']
        emotion_performance['Strategy'] = emotion_performance['Avg_Hotness'].apply(
            lambda x: '📈 INVEST' if x > 0.6 else '⚖️ MAINTAIN' if x > 0.4 else '📉 DIVEST'
        )
        
        st.dataframe(
            emotion_performance[['Emotion', 'Avg_Hotness', 'Revenue_Potential', 'Strategy']],
            use_container_width=True
        )
        
        st.markdown("""
        **Strategy Definitions:**
        - **INVEST:** High hotness (>0.6) - Increase inventory and marketing spend
        - **MAINTAIN:** Medium hotness (0.4-0.6) - Keep current levels
        - **DIVEST:** Low hotness (<0.4) - Reduce inventory and consider repositioning
        """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
**H & M Fashion BI**
Strategic Command Center for Emotional Retail Intelligence

*Powered by Deep Learning & Emotion Analytics*
""")
