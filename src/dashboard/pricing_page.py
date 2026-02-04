"""
Dynamic Pricing Dashboard Page: owner recommendations, renter alerts, investor scoring, current prices.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pricing import PricingEngine, PriceConstraints, MarketPosition
from pricing.market_analyzer import MarketAnalyzer
from api.utils import load_model, prepare_features
from training.config import MLFLOW_TRACKING_URI
import os
import mlflow

# Chart color theme (matching main dashboard)
CHART_COLORS = ["#2872A1", "#CBDDE9", "#5a9bc4", "#9ec9e0", "#1e5a82"]


def render_pricing_page():
    """Render the complete dynamic pricing dashboard page."""
    
    # Initialize pricing engine and analyzer
    try:
        engine = PricingEngine()
        analyzer = MarketAnalyzer()
    except Exception as e:
        st.error(f"Pricing engine not available: {e}")
        st.info("Ensure market signals database is populated: `python -m src.data.generate_market_signals`")
        return
    
    # Load ML model for base price prediction (use session state to cache)
    if 'pricing_model' not in st.session_state:
        try:
            # Set MLflow tracking URI to match training config (absolute path)
            # Set env var so load_model uses it
            os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = os.getenv("MLFLOW_MODEL_URI")
            run_id = os.getenv("MLFLOW_RUN_ID")
            # load_model will auto-detect latest run if both are None
            st.session_state.pricing_model = load_model(model_uri=model_uri, run_id=run_id)
        except Exception as e:
            st.warning(f"ML model not available for base price prediction: {e}")
            st.info("Train a model first: `python -m src.training.train` or ensure MLFLOW_MODEL_URI/RUN_ID is set.")
            st.session_state.pricing_model = None
    
    model = st.session_state.get('pricing_model')
    
    st.header("Dynamic Pricing")
    st.markdown("Get pricing recommendations, alerts, and market insights based on real-time market conditions.")
    
    # Tabs for different pricing use cases
    pricing_tab1, pricing_tab2, pricing_tab3, pricing_tab4, pricing_tab5 = st.tabs([
        "Owner Recommendations",
        "Renter Alerts",
        "Investor Opportunities",
        "Current Dynamic Price",
        "Market Signals"
    ])
    
    # Tab 1: Owner Recommendations
    with pricing_tab1:
        st.subheader("Owner Pricing Recommendations")
        st.markdown("Get recommended list price based on market conditions and your pricing strategy.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Property Details**")
            area_sqft = st.number_input("Area (sqft)", min_value=100.0, max_value=10000.0, value=1500.0, step=50.0)
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=10, step=1)
            
            property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo", "Townhouse"])
            location = st.selectbox("Location", ["Downtown", "Urban", "Suburbs", "Rural"])
            
            has_parking = st.checkbox("Has Parking", value=True)
            has_gym = st.checkbox("Has Gym", value=False)
            has_pool = st.checkbox("Has Pool", value=False)
        
        with col2:
            st.markdown("**Pricing Strategy**")
            market_position = st.selectbox(
                "Market Position",
                ["market", "aggressive", "conservative"],
                help="Market: at market rate. Aggressive: list above market. Conservative: list below for faster sale."
            )
            min_price = st.number_input("Min Price (optional)", min_value=0.0, value=0.0, step=10000.0)
            max_price = st.number_input("Max Price (optional)", min_value=0.0, value=0.0, step=10000.0)
            as_of_date = st.date_input("Market Date (optional)", value=None, help="Leave empty for latest market conditions")
            
            if st.button("Get Recommendation"):
                if model is None:
                    st.error("ML model not loaded. Cannot compute base price.")
                else:
                    try:
                        # Build property dict
                        prop_dict = {
                            "area_sqft": area_sqft,
                            "bedrooms": bedrooms,
                            "bathrooms": bathrooms,
                            "age": age,
                            "has_parking": 1 if has_parking else 0,
                            "has_gym": 1 if has_gym else 0,
                            "has_pool": 1 if has_pool else 0,
                            "property_type_" + property_type: 1,
                            "location_" + location: 1,
                        }
                        for pt in ["Apartment", "Condo", "House", "Townhouse"]:
                            if pt != property_type:
                                prop_dict[f"property_type_{pt}"] = 0
                        for loc in ["Downtown", "Urban", "Suburbs", "Rural"]:
                            if loc != location:
                                prop_dict[f"location_{loc}"] = 0
                        
                        # Get base price
                        features_df = prepare_features(prop_dict)
                        base_price = float(model.predict(features_df)[0])
                        
                        # Get recommendation
                        mp = MarketPosition.MARKET
                        if market_position == "aggressive":
                            mp = MarketPosition.AGGRESSIVE
                        elif market_position == "conservative":
                            mp = MarketPosition.CONSERVATIVE
                        
                        constraints = PriceConstraints(
                            min_price=min_price if min_price > 0 else None,
                            max_price=max_price if max_price > 0 else None,
                            market_position=mp,
                        )
                        
                        rec = engine.recommend_for_owner(
                            base_price=base_price,
                            location=location,
                            constraints=constraints,
                            as_of_date=as_of_date,
                        )
                        
                        # Display results
                        st.success("Recommendation Generated")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Recommended Price", f"${rec.recommended_price:,.0f}")
                        col_b.metric("Price Range", f"${rec.price_min:,.0f} - ${rec.price_max:,.0f}")
                        col_c.metric("Base Price", f"${base_price:,.0f}")
                        
                        st.info(f"**Reasoning:** {rec.reasoning}")
                        st.caption(f"Demand Level: {rec.demand_level} | Market Position: {rec.market_position_used}")
                        
                    except Exception as e:
                        st.error(f"Error generating recommendation: {e}")
    
    # Tab 2: Renter Alerts
    with pricing_tab2:
        st.subheader("Renter Price Alerts")
        st.markdown("Check if an asking price is fair based on current market conditions.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Property Details**")
            area_sqft_r = st.number_input("Area (sqft)", min_value=100.0, max_value=10000.0, value=1500.0, step=50.0, key="r_area")
            bedrooms_r = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1, key="r_bed")
            bathrooms_r = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key="r_bath")
            age_r = st.number_input("Property Age (years)", min_value=0, max_value=100, value=10, step=1, key="r_age")
            
            property_type_r = st.selectbox("Property Type", ["House", "Apartment", "Condo", "Townhouse"], key="r_type")
            location_r = st.selectbox("Location", ["Downtown", "Urban", "Suburbs", "Rural"], key="r_loc")
            
            has_parking_r = st.checkbox("Has Parking", value=True, key="r_park")
            has_gym_r = st.checkbox("Has Gym", value=False, key="r_gym")
            has_pool_r = st.checkbox("Has Pool", value=False, key="r_pool")
        
        with col2:
            st.markdown("**Price Check**")
            asking_price = st.number_input("Asking Price", min_value=0.0, value=400000.0, step=10000.0)
            fair_band_pct = st.slider("Fair Band (%)", min_value=0.0, max_value=20.0, value=8.0, step=0.5) / 100.0
            as_of_date_r = st.date_input("Market Date (optional)", value=None, key="r_date")
            
            if st.button("Check Price", key="r_btn"):
                if model is None:
                    st.error("ML model not loaded. Cannot compute base price.")
                else:
                    try:
                        prop_dict_r = {
                            "area_sqft": area_sqft_r,
                            "bedrooms": bedrooms_r,
                            "bathrooms": bathrooms_r,
                            "age": age_r,
                            "has_parking": 1 if has_parking_r else 0,
                            "has_gym": 1 if has_gym_r else 0,
                            "has_pool": 1 if has_pool_r else 0,
                            "property_type_" + property_type_r: 1,
                            "location_" + location_r: 1,
                        }
                        for pt in ["Apartment", "Condo", "House", "Townhouse"]:
                            if pt != property_type_r:
                                prop_dict_r[f"property_type_{pt}"] = 0
                        for loc in ["Downtown", "Urban", "Suburbs", "Rural"]:
                            if loc != location_r:
                                prop_dict_r[f"location_{loc}"] = 0
                        
                        features_df_r = prepare_features(prop_dict_r)
                        base_price_r = float(model.predict(features_df_r)[0])
                        
                        alert = engine.alert_for_renter(
                            asking_price=asking_price,
                            base_price=base_price_r,
                            location=location_r,
                            as_of_date=as_of_date_r,
                            fair_band_pct=fair_band_pct,
                        )
                        
                        if alert.is_fair:
                            st.success(alert.message)
                        else:
                            st.warning(alert.message)
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Asking Price", f"${alert.asking_price:,.0f}")
                        col_b.metric("Fair Range", f"${alert.fair_low:,.0f} - ${alert.fair_high:,.0f}")
                        col_c.metric("Base Price", f"${alert.base_price:,.0f}")
                        
                        # Visual indicator
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=["Fair Low", "Asking", "Fair High"],
                            y=[alert.fair_low, alert.asking_price, alert.fair_high],
                            marker_color=[CHART_COLORS[0], CHART_COLORS[2] if alert.is_fair else CHART_COLORS[4], CHART_COLORS[0]]
                        ))
                        fig.update_layout(title="Price Comparison", yaxis_title="Price ($)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error checking price: {e}")
    
    # Tab 3: Investor Opportunities
    with pricing_tab3:
        st.subheader("Investor Opportunity Scoring")
        st.markdown("Score investment opportunities and get suggested bid prices.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Property Details**")
            area_sqft_i = st.number_input("Area (sqft)", min_value=100.0, max_value=10000.0, value=1500.0, step=50.0, key="i_area")
            bedrooms_i = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1, key="i_bed")
            bathrooms_i = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key="i_bath")
            age_i = st.number_input("Property Age (years)", min_value=0, max_value=100, value=10, step=1, key="i_age")
            
            property_type_i = st.selectbox("Property Type", ["House", "Apartment", "Condo", "Townhouse"], key="i_type")
            location_i = st.selectbox("Location", ["Downtown", "Urban", "Suburbs", "Rural"], key="i_loc")
            
            has_parking_i = st.checkbox("Has Parking", value=True, key="i_park")
            has_gym_i = st.checkbox("Has Gym", value=False, key="i_gym")
            has_pool_i = st.checkbox("Has Pool", value=False, key="i_pool")
        
        with col2:
            st.markdown("**Investment Parameters**")
            min_roi_pct = st.number_input("Min ROI Target (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)
            list_discount_pct = st.slider("Suggested Discount (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 100.0
            as_of_date_i = st.date_input("Market Date (optional)", value=None, key="i_date")
            
            if st.button("Score Opportunity", key="i_btn"):
                if model is None:
                    st.error("ML model not loaded. Cannot compute base price.")
                else:
                    try:
                        prop_dict_i = {
                            "area_sqft": area_sqft_i,
                            "bedrooms": bedrooms_i,
                            "bathrooms": bathrooms_i,
                            "age": age_i,
                            "has_parking": 1 if has_parking_i else 0,
                            "has_gym": 1 if has_gym_i else 0,
                            "has_pool": 1 if has_pool_i else 0,
                            "property_type_" + property_type_i: 1,
                            "location_" + location_i: 1,
                        }
                        for pt in ["Apartment", "Condo", "House", "Townhouse"]:
                            if pt != property_type_i:
                                prop_dict_i[f"property_type_{pt}"] = 0
                        for loc in ["Downtown", "Urban", "Suburbs", "Rural"]:
                            if loc != location_i:
                                prop_dict_i[f"location_{loc}"] = 0
                        
                        features_df_i = prepare_features(prop_dict_i)
                        base_price_i = float(model.predict(features_df_i)[0])
                        
                        opp = engine.opportunity_for_investor(
                            base_price=base_price_i,
                            location=location_i,
                            as_of_date=as_of_date_i,
                            min_roi_pct=min_roi_pct,
                            list_discount_pct=list_discount_pct,
                        )
                        
                        # Display score
                        score_color = CHART_COLORS[0] if opp.score >= 75 else CHART_COLORS[2] if opp.score >= 50 else CHART_COLORS[4]
                        st.metric("Opportunity Score", f"{opp.score:.1f}/100", delta=f"{'Meets ROI' if opp.meets_roi else 'Below ROI target'}")
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Suggested Bid", f"${opp.suggested_bid:,.0f}")
                        col_b.metric("Expected Value", f"${opp.expected_value:,.0f}")
                        col_c.metric("Base Price", f"${base_price_i:,.0f}")
                        
                        st.info(f"**Reasoning:** {opp.reasoning}")
                        
                        # Score visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=opp.score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Opportunity Score"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': score_color},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 75], 'color': "gray"},
                                    {'range': [75, 100], 'color': "darkgray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': min_roi_pct * 10
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error scoring opportunity: {e}")
    
    # Tab 4: Current Dynamic Price
    with pricing_tab4:
        st.subheader("Current Dynamic Price")
        st.markdown("Get real-time surge-style pricing based on current market demand.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Property Details**")
            area_sqft_c = st.number_input("Area (sqft)", min_value=100.0, max_value=10000.0, value=1500.0, step=50.0, key="c_area")
            bedrooms_c = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1, key="c_bed")
            bathrooms_c = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key="c_bath")
            age_c = st.number_input("Property Age (years)", min_value=0, max_value=100, value=10, step=1, key="c_age")
            
            property_type_c = st.selectbox("Property Type", ["House", "Apartment", "Condo", "Townhouse"], key="c_type")
            location_c = st.selectbox("Location", ["Downtown", "Urban", "Suburbs", "Rural"], key="c_loc")
            
            has_parking_c = st.checkbox("Has Parking", value=True, key="c_park")
            has_gym_c = st.checkbox("Has Gym", value=False, key="c_gym")
            has_pool_c = st.checkbox("Has Pool", value=False, key="c_pool")
        
        with col2:
            st.markdown("**Market Date**")
            as_of_date_c = st.date_input("Market Date (optional)", value=None, key="c_date")
            
            if st.button("Get Current Price", key="c_btn"):
                if model is None:
                    st.error("ML model not loaded. Cannot compute base price.")
                else:
                    try:
                        prop_dict_c = {
                            "area_sqft": area_sqft_c,
                            "bedrooms": bedrooms_c,
                            "bathrooms": bathrooms_c,
                            "age": age_c,
                            "has_parking": 1 if has_parking_c else 0,
                            "has_gym": 1 if has_gym_c else 0,
                            "has_pool": 1 if has_pool_c else 0,
                            "property_type_" + property_type_c: 1,
                            "location_" + location_c: 1,
                        }
                        for pt in ["Apartment", "Condo", "House", "Townhouse"]:
                            if pt != property_type_c:
                                prop_dict_c[f"property_type_{pt}"] = 0
                        for loc in ["Downtown", "Urban", "Suburbs", "Rural"]:
                            if loc != location_c:
                                prop_dict_c[f"location_{loc}"] = 0
                        
                        features_df_c = prepare_features(prop_dict_c)
                        base_price_c = float(model.predict(features_df_c)[0])
                        
                        current = engine.current_dynamic_price(
                            base_price=base_price_c,
                            location=location_c,
                            as_of_date=as_of_date_c,
                        )
                        
                        st.success("Current Price Calculated")
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Current Price", f"${current['current_price']:,.0f}")
                        col_b.metric("Base Price", f"${current['base_price']:,.0f}")
                        col_c.metric("Demand Multiplier", f"{current['demand_multiplier']:.2f}x")
                        col_d.metric("Competition Effect", f"{current['competition_effect']:.2f}x")
                        
                        st.caption(f"Demand Level: {current['demand_level']}")
                        
                        # Price comparison chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=["Base Price", "Current Dynamic Price"],
                            y=[current['base_price'], current['current_price']],
                            marker_color=CHART_COLORS,
                            text=[f"${current['base_price']:,.0f}", f"${current['current_price']:,.0f}"],
                            textposition="outside"
                        ))
                        fig.update_layout(title="Base vs Current Dynamic Price", yaxis_title="Price ($)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error getting current price: {e}")
    
    # Tab 5: Market Signals Visualization
    with pricing_tab5:
        st.subheader("Market Signals Visualization")
        st.markdown("View demand, competition, and seasonality trends over time.")
        
        try:
            latest_date = analyzer.get_latest_signal_date()
            if latest_date is None:
                st.warning("No market signals data available. Run: `python -m src.data.generate_market_signals`")
            else:
                st.caption(f"Latest signal date: {latest_date}")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    selected_location = st.selectbox("Location", ["All", "Downtown", "Urban", "Suburbs", "Rural"])
                with col2:
                    days_back = st.slider("Days of History", min_value=30, max_value=365, value=90, step=30)
                
                end_date = latest_date
                start_date = end_date - timedelta(days=days_back)
                
                location_filter = None if selected_location == "All" else selected_location
                history_df = analyzer.get_history(start_date, end_date, location=location_filter)
                
                if not history_df.empty:
                    history_df['signal_date'] = pd.to_datetime(history_df['signal_date'])
                    
                    # Demand multiplier over time
                    st.markdown("**Demand Multiplier Over Time**")
                    fig_demand = px.line(
                        history_df,
                        x='signal_date',
                        y='demand_multiplier',
                        color='location' if selected_location == "All" else None,
                        title='Demand Multiplier Trend',
                        labels={'demand_multiplier': 'Demand Multiplier', 'signal_date': 'Date'}
                    )
                    fig_demand.update_layout(colorway=CHART_COLORS)
                    st.plotly_chart(fig_demand, use_container_width=True)
                    
                    # Competition listings over time
                    st.markdown("**Competition (Active Listings) Over Time**")
                    fig_comp = px.line(
                        history_df,
                        x='signal_date',
                        y='competition_listings',
                        color='location' if selected_location == "All" else None,
                        title='Competition Trend',
                        labels={'competition_listings': 'Active Listings', 'signal_date': 'Date'}
                    )
                    fig_comp.update_layout(colorway=CHART_COLORS)
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Seasonality by month
                    st.markdown("**Seasonality by Month**")
                    seasonality_df = analyzer.get_seasonality()
                    if not seasonality_df.empty:
                        fig_season = px.bar(
                            seasonality_df,
                            x='month',
                            y='seasonality_factor',
                            title='Average Seasonality Factor by Month',
                            labels={'seasonality_factor': 'Seasonality Factor', 'month': 'Month'}
                        )
                        fig_season.update_traces(marker_color=CHART_COLORS[0])
                        st.plotly_chart(fig_season, use_container_width=True)
                    
                    # Summary stats
                    st.markdown("**Summary Statistics**")
                    summary = history_df.groupby('location' if selected_location == "All" else None).agg({
                        'demand_multiplier': ['mean', 'min', 'max'],
                        'competition_listings': ['mean', 'min', 'max'],
                    }).round(2)
                    st.dataframe(summary)
                else:
                    st.info("No data available for selected period.")
        except Exception as e:
            st.error(f"Error loading market signals: {e}")
