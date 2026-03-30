import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pricing Strategy Recommender",
    page_icon="🏷️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F1923;
    }
    [data-testid="stSidebar"] * {
        color: #E8E8E4 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #A0A89A !important;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #F7F6F2;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #E8E4DC;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #888 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem !important;
        color: #1A1A1A !important;
    }

    /* Divider */
    hr { border-color: #E8E4DC; }

    /* Button */
    .stButton > button {
        background: #C84B31 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em;
        height: 48px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* Info / success boxes */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('app_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

artifacts         = load_artifacts()
model             = artifacts['model']
le                = artifacts['label_encoder']
strategy_advice   = artifacts['strategy_advice']
best_per_category = artifacts['best_per_category']
category_list     = artifacts['category_list']
df_stats          = artifacts['df_stats']

# ── Strategy config — friendlier labels ──────────────────
STRATEGY_DISPLAY = {
    'Smart Discounter':          'Smart Discounter',
    'Overpriced Low Performer':  'High-Price Risk',
    'Underperformer':            'Needs Optimisation',
}

strategy_config = {
    'Smart Discounter': {
        'icon':  '🏆',
        'color': '#1A3A5C',
        'bg':    '#E8F0FE',
        'badge': '#2ECC71',
    },
    'Overpriced Low Performer': {
        'icon':  '📉',          # changed from 💎 — makes more sense
        'color': '#C0392B',
        'bg':    '#FDECEA',
        'badge': '#E74C3C',
    },
    'Underperformer': {
        'icon':  '🔧',          # "fixable" framing
        'color': '#7F6A00',
        'bg':    '#FFFBEA',
        'badge': '#F39C12',
    },
}


# ── Predict ───────────────────────────────────────────────
def predict_strategy(category_name, price, rating, shipping_cost):
    cat_data = df_stats[df_stats['category_name'] == category_name]

    if len(cat_data) == 0:
        cat_avg_price = df_stats['price'].mean()
        cat_id        = 0
    else:
        cat_avg_price = cat_data['price'].mean()
        cat_id        = cat_data['category_id'].iloc[0]

    effective_price       = price * (1 - 0.30)
    price_vs_category_avg = price / cat_avg_price if cat_avg_price > 0 else 1
    price_rank            = (cat_data['price'] < price).mean() \
                            if len(cat_data) > 0 else 0.5

    features = pd.DataFrame([{
        'price':                  price,
        'shippingCost':           shipping_cost,
        'rating':                 rating,
        'category_id':            cat_id,
        'price_rank_in_category': price_rank,
        'price_vs_category_avg':  price_vs_category_avg,
        'effective_price':        effective_price,
    }])

    prediction    = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    strategy_name = le.inverse_transform([prediction])[0]
    confidence    = probabilities.max() * 100

    advice   = strategy_advice[strategy_name]
    cat_best = best_per_category[
        best_per_category['category_name'] == category_name
    ]
    category_winner   = cat_best['best_strategy'].values[0] \
                        if len(cat_best) > 0 else 'Smart Discounter'
    category_avg_sold = cat_best['avg_sold'].values[0] \
                        if len(cat_best) > 0 else 0

    # Category-aware price thresholds (no more magic number 200)
    price_pct25 = cat_data['price'].quantile(0.25) \
                  if len(cat_data) > 0 else df_stats['price'].quantile(0.25)
    price_pct75 = cat_data['price'].quantile(0.75) \
                  if len(cat_data) > 0 else df_stats['price'].quantile(0.75)

    return {
        'strategy':          strategy_name,
        'display_name':      STRATEGY_DISPLAY[strategy_name],
        'confidence':        confidence,
        'discount_range':    advice['discount_range'],
        'shipping_tip':      advice['shipping'],
        'pricing_tip':       advice['pricing'],
        'key_action':        advice['key_action'],
        'category_winner':   category_winner,
        'category_avg_sold': category_avg_sold,
        'probabilities':     {
            cls: prob * 100
            for cls, prob in zip(le.classes_, probabilities)
        },
        'price_pct25': price_pct25,
        'price_pct75': price_pct75,
    }


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 Product Details")
    st.caption("Fill in your product information:")

    category = st.selectbox(
        "Product Category",
        options=category_list,
        index=category_list.index('consumer-electronics')
        if 'consumer-electronics' in category_list else 0
    )

    price = st.number_input(
        "Price (DT)",
        min_value=1.0,
        max_value=10_000.0,
        value=100.0,
        step=10.0
    )

    rating = st.slider(
        "Product Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    shipping = st.number_input(
        "Shipping Cost (DT)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
        help="Enter 0 for free shipping"
    )

    st.divider()

    # Live input sanity hints — no need to hit the button first
    hints = []
    if rating < 3.0:
        hints.append("⚠️ Low rating may hurt performance")
    if shipping > 20:
        hints.append("⚠️ High shipping cost reduces conversions")
    if hints:
        for h in hints:
            st.warning(h, icon=None)

    predict_btn = st.button(
        "🚀 Get Strategy",
        use_container_width=True,
        type="primary"
    )


# ── Header ────────────────────────────────────────────────
st.markdown("# 🏷️ Pricing Strategy Recommender")
st.markdown(
    "> Enter your product details to get an ML-powered pricing strategy recommendation."
)
st.divider()


# ── Results ───────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analysing pricing strategy…"):
        report = predict_strategy(category, price, rating, shipping)

    cfg          = strategy_config[report['strategy']]
    display_name = report['display_name']

    # Banner
    st.markdown(f"""
    <div style="
        background:{cfg['bg']};
        border-left:6px solid {cfg['color']};
        padding:22px 24px;
        border-radius:10px;
        margin-bottom:8px;">
        <h2 style="color:{cfg['color']};margin:0;font-family:'DM Serif Display',serif;">
            {cfg['icon']} {display_name}
        </h2>
        <p style="color:#666;margin:6px 0 0;">
            Model confidence: <strong>{report['confidence']:.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Explain the confidence gap if winner ≠ predicted strategy
    if report['category_winner'] != report['strategy']:
        winner_prob = report['probabilities'].get(report['category_winner'], 0)
        st.caption(
            f"ℹ️ The category winner **{STRATEGY_DISPLAY[report['category_winner']]}** "
            f"has {winner_prob:.1f}% model probability for your current inputs — "
            f"adjust price / shipping to move toward it."
        )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommended Discount", report['discount_range'])
    col2.metric("Confidence",           f"{report['confidence']:.1f}%")
    col3.metric("Category Winner",      STRATEGY_DISPLAY.get(
        report['category_winner'], report['category_winner'].split()[0]
    ))
    col4.metric("Category Avg Sold",    f"{report['category_avg_sold']:.0f} units")

    st.divider()

    left, right = st.columns(2)

    # ── Left column ───────────────────────────────────────
    with left:
        st.subheader("📋 Recommendations")
        st.info(f"**💰 Pricing:** {report['pricing_tip']}")
        st.info(f"**🚚 Shipping:** {report['shipping_tip']}")
        st.success(f"**⚡ Key Action:** {report['key_action']}")

        st.subheader("📊 Category Insights")
        st.markdown(f"""
| Metric | Value |
|---|---|
| Category | `{category}` |
| Winning Strategy | **{STRATEGY_DISPLAY.get(report['category_winner'], report['category_winner'])}** |
| Avg Units Sold (winner) | **{report['category_avg_sold']:.0f}** |
| 25th–75th Price Percentile | **{report['price_pct25']:.0f} – {report['price_pct75']:.0f} DT** |
        """)

    # ── Right column ──────────────────────────────────────
    with right:
        st.subheader("📈 Strategy Probability")

        probs = report['probabilities']
        # Use display names in chart
        display_keys = [STRATEGY_DISPLAY.get(k, k) for k in probs.keys()]
        colors       = [strategy_config[s]['color'] for s in probs.keys()]

        fig = go.Figure(go.Bar(
            x            = list(probs.values()),
            y            = display_keys,
            orientation  = 'h',
            marker_color = colors,
            text         = [f"{v:.1f}%" for v in probs.values()],
            textposition = 'outside',
        ))
        fig.update_layout(
            title       = 'Model Confidence per Strategy',
            xaxis_title = 'Probability (%)',
            height      = 280,
            margin      = dict(l=0, r=60, t=40, b=0),
            xaxis       = dict(range=[0, 115]),
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor= 'rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Your Product vs Ideal Profile")

        radar_cats = ['Price', 'Rating', 'Shipping', 'Value']
        max_price  = df_stats['price'].quantile(0.99)

        your_vals = [
            1 - min(price / max_price, 1),
            rating / 5,
            1 - min(shipping / 100, 1),
            (rating / 5) * (1 - min(price / max_price, 1)),
        ]
        ideal_vals = [0.8, 0.95, 1.0, 0.8]

        def hex_to_rgba(hex_color, alpha=0.2):
            """Convert #RRGGBB to rgba(r,g,b,alpha) for Plotly."""
            h = hex_color.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f'rgba({r},{g},{b},{alpha})'

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r          = your_vals + [your_vals[0]],
            theta      = radar_cats + [radar_cats[0]],
            fill       = 'toself',
            name       = 'Your Product',
            line_color = cfg['color'],
            fillcolor  = hex_to_rgba(cfg['color'], 0.2),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r          = ideal_vals + [ideal_vals[0]],
            theta      = radar_cats + [radar_cats[0]],
            fill       = 'toself',
            name       = 'Ideal (Smart Discounter)',
            line_color = '#2ECC71',
            fillcolor  = hex_to_rgba('#2ECC71', 0.2),
        ))
        fig_radar.update_layout(
            polar  = dict(radialaxis=dict(visible=True, range=[0, 1])),
            height = 340,
            margin = dict(l=40, r=40, t=20, b=40),
            legend = dict(orientation='h', y=-0.1),
            plot_bgcolor  = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── Action Plan — category-aware thresholds ───────────
    st.subheader("🗺️ Action Plan to Become Smart Discounter")

    p25, p75 = report['price_pct25'], report['price_pct75']
    price_ok   = p25 <= price <= p75
    rating_ok  = rating >= 4.0
    shipping_ok = shipping == 0

    step1, step2, step3 = st.columns(3)

    with step1:
        price_status = "✅ In the sweet spot" if price_ok else \
                       ("⬇️ Above category median — consider reducing" if price > p75
                        else "⬆️ Below median — room to raise price")
        st.markdown(f"""
**Step 1 — Price** 💰

Current: **{price:.2f} DT**

Category range: *{p25:.0f} – {p75:.0f} DT*

{price_status}
        """)

    with step2:
        rating_status = "✅ Great rating" if rating_ok else \
                        f"⚠️ Below 4.0 — focus on product quality ({4.0 - rating:.1f} pts gap)"
        st.markdown(f"""
**Step 2 — Rating** ⭐

Current: **{rating:.1f} / 5**

{rating_status}
        """)

    with step3:
        shipping_status = "✅ Free shipping!" if shipping_ok else \
                          f"⚠️ Consider free shipping (saves buyer {shipping:.2f} DT)"
        st.markdown(f"""
**Step 3 — Shipping** 🚚

Current: **{shipping:.2f} DT**

{shipping_status}
        """)


# ── Default (no prediction yet) ───────────────────────────
else:
    st.markdown("### 👈 Fill in your product details and click **Get Strategy**")

    st.subheader("📊 Strategy Overview")

    col1, col2, col3 = st.columns(3)

    overviews = [
        ('Smart Discounter',         'Smart Discounter',
         '🏆', '#1A3A5C', '#E8F0FE',
         'Affordable price + smart discount + great rating.',
         '**Wins in 99% of categories**'),
        ('Overpriced Low Performer', 'High-Price Risk',
         '📉', '#C0392B', '#FDECEA',
         'Very high price, low sales, poor rating.',
         '**Avoid this pattern**'),
        ('Underperformer',           'Needs Optimisation',
         '🔧', '#7F6A00', '#FFFBEA',
         'Inconsistent pricing, mediocre rating, low sales.',
         '**Fixable with the right changes**'),
    ]

    for col, (_, label, icon, color, bg, desc, cta) in zip(
        [col1, col2, col3], overviews
    ):
        with col:
            st.markdown(f"""
<div style="background:{bg};padding:20px;border-radius:10px;
            border-left:4px solid {color};height:100%;">
    <h4 style="color:{color};margin:0 0 8px;">{icon} {label}</h4>
    <p style="margin:0 0 10px;font-size:0.9rem;">{desc}</p>
    <span style="font-weight:600;font-size:0.85rem;">{cta}</span>
</div>
            """, unsafe_allow_html=True)

    st.divider()
    st.info(
        "💡 **Pro Tip:** Start with a product in the *consumer-electronics* "
        "category for the best results!"
    )