import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb  # <-- Import XGBoost
import warnings
warnings.filterwarnings('ignore')

# --- Set page configuration ---
st.set_page_config(
    page_title="PGA Tour Performance Analytics",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and description ---
st.title("PGA Tour Performance Analytics")
st.markdown("""
This application analyzes professional golf performance metrics and their relationship to earnings.
Explore different visualizations, player clustering, and predictive models to understand key factors
influencing golfer performance and financial success.
""")

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Data Overview", "Exploratory Analysis", "Performance Clustering", "Earnings Prediction"]
)

# --- Function to load data ---
@st.cache_data
def load_data():
    try:
        # Make sure 'golf_data.csv' is in the same directory as your script
        # or provide the full path.
        df = pd.read_csv('golf_data.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'golf_data.csv' not found. Please place the data file in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Load the data ---
df = load_data()

if df is not None:
    # --- Clean the data ---
    def clean_data(df):
        df_clean = df.copy()
        key_columns = ['sg_total', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g']
        df_clean = df_clean.dropna(subset=key_columns)
        df_clean['pos'] = pd.to_numeric(df_clean['pos'], errors='coerce')
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])
        numerical_cols = df_clean.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            if df_clean[col].isnull().any(): # Check if there are NaNs before filling
                 df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        return df_clean

    # Clean the data
    df_clean = clean_data(df)

    # --- Data Overview Page ---
    if page == "Data Overview":
        st.header("Dataset Overview")
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}")
        st.subheader("Sample Data")
        st.dataframe(df_clean.head())
        st.subheader("Column Descriptions")
        column_descriptions = {
            'Player_initial_last': 'Player name (initial + last name)', 'tournament id': 'Unique identifier for the tournament',
            'player id': 'Unique identifier for the player', 'hole_par': 'Par value for the hole',
            'strokes': 'Number of strokes taken', 'made_cut': 'Whether the player made the cut (1) or not (0)',
            'pos': 'Final position in the tournament', 'player': 'Full player name',
            'tournament_name': 'Name of the tournament', 'course': 'Course name', 'date': 'Tournament date',
            'purse': 'Tournament prize money (in USD)', 'season': 'Season year',
            'no_cut': 'Whether the tournament has no cut (1) or has a cut (0)', 'Finish': 'Finish position as string (including ties)',
            'sg_putt': 'Strokes gained: putting', 'sg_arg': 'Strokes gained: around the green',
            'sg_app': 'Strokes gained: approach', 'sg_ott': 'Strokes gained: off the tee',
            'sg_t2g': 'Strokes gained: tee to green', 'sg_total': 'Strokes gained: total'
        }
        for col, desc in column_descriptions.items():
            if col in df_clean.columns:
                st.write(f"**{col}**: {desc}")
        st.subheader("Missing Values Analysis (After Cleaning)")
        missing_vals = df_clean.isnull().sum()
        missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)
        if not missing_vals.empty:
            st.write("Columns with remaining missing values:")
            st.write(missing_vals)
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_vals.plot(kind='bar', ax=ax)
            plt.title('Missing Values by Column (After Cleaning)')
            plt.ylabel('Count'); plt.xlabel('Columns'); plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No missing values remaining in the cleaned dataset.")

    # --- Exploratory Analysis Page ---
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        st.subheader("Strokes Gained Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_clean, x='sg_total', title='Distribution of Strokes Gained: Total', labels={'sg_total': 'Strokes Gained: Total'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sg_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott']
            sg_data = df_clean[sg_cols].mean().reset_index()
            sg_data.columns = ['Category', 'Average Strokes Gained']
            fig = px.bar(sg_data, x='Category', y='Average Strokes Gained', title='Average Strokes Gained by Category', labels={'Category': 'Strokes Gained Category', 'Average Strokes Gained': 'Average Value'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Analysis")
        numeric_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'pos', 'purse']
        # Ensure all numeric_cols exist before calculating correlation
        valid_numeric_cols = [col for col in numeric_cols if col in df_clean.columns]
        if len(valid_numeric_cols) > 1: # Need at least 2 columns for correlation
            corr_df = df_clean[valid_numeric_cols].corr()
            fig = px.imshow(corr_df, title='Correlation Matrix of Performance Metrics', labels=dict(color="Correlation"), x=corr_df.columns, y=corr_df.columns, color_continuous_scale='RdBu_r', text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns available for correlation analysis.")


        st.subheader("Position vs. Strokes Gained")
        sg_position_option = st.selectbox('Select Strokes Gained Category', ['sg_total', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g'])
        if 'pos' in df_clean.columns and sg_position_option in df_clean.columns:
            filtered_df = df_clean[df_clean['pos'].notna() & (df_clean['pos'] <= 100)] # Ensure pos is not NaN
            fig = px.scatter(filtered_df, x=sg_position_option, y='pos', title=f'Tournament Position vs. {sg_position_option}', labels={sg_position_option: sg_position_option.replace('_', ' ').title(), 'pos': 'Tournament Position'}, trendline='ols')
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.warning(f"Column 'pos' or '{sg_position_option}' not found or invalid for plotting.")


        required_cols_eda = ['tournament_name', 'purse']
        missing_cols_eda = [col for col in required_cols_eda if col not in df_clean.columns]
        if not missing_cols_eda:
             st.subheader("Tournament Purse Analysis")
             top_tournaments = df_clean.groupby('tournament_name')['purse'].mean().nlargest(10) # Use nlargest for clarity
             if not top_tournaments.empty:
                 fig = px.bar(top_tournaments, title='Top 10 Tournaments by Average Prize Money', labels={'value': 'Average Purse (USD)', 'tournament_name': 'Tournament'})
                 st.plotly_chart(fig, use_container_width=True)
             else:
                 st.warning("No tournament data available for purse analysis.")
        else:
             st.error(f"Missing columns for purse analysis: {', '.join(missing_cols_eda)}")


    # --- Performance Clustering Page ---
    elif page == "Performance Clustering":
        st.header("Player Performance Clustering")
        st.markdown("""
        This analysis uses K-means clustering to group golfers based on their performance patterns (average Strokes Gained metrics).
        """)
        clustering_features = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott'] # Define features used
        required_cols_cluster = ['player'] + clustering_features + ['sg_t2g', 'sg_total']
        missing_cols_cluster = [col for col in required_cols_cluster if col not in df_clean.columns]

        if not missing_cols_cluster:
            player_stats = df_clean.groupby('player')[required_cols_cluster[1:]].mean().reset_index() # Use required cols
            player_stats = player_stats.dropna(subset=clustering_features) # Drop NaN based on features used

            if not player_stats.empty:
                X = player_stats[clustering_features]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                st.subheader("Optimal Number of Clusters (Elbow Method)")
                inertia = []
                k_range = range(1, 11)
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') # Use n_init='auto'
                    kmeans.fit(X_scaled)
                    inertia.append(kmeans.inertia_)
                fig = px.line(x=k_range, y=inertia, markers=True, title='Elbow Method for Optimal k', labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'})
                st.plotly_chart(fig, use_container_width=True)

                num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                player_stats['cluster'] = kmeans.fit_predict(X_scaled)

                st.subheader("Cluster Centers (Average Strokes Gained)")
                cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=clustering_features)
                cluster_centers.index = [f"Cluster {i}" for i in range(num_clusters)]
                st.dataframe(cluster_centers.style.format("{:.3f}")) # Format for better readability

                st.subheader("Cluster Visualization")
                x_axis = st.selectbox('Select X-axis feature', clustering_features, index=2)
                y_axis = st.selectbox('Select Y-axis feature', clustering_features, index=0)
                fig = px.scatter(player_stats, x=x_axis, y=y_axis, color='cluster', title=f'Player Clustering by {x_axis} and {y_axis}', labels={x_axis: x_axis.replace('_', ' ').title(), y_axis: y_axis.replace('_', ' ').title()}, hover_data=['player', 'sg_total'])
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Cluster Profiles (Radar Chart)")
                fig = go.Figure()
                radar_features = clustering_features # Use the same features as clustering
                min_val = cluster_centers.values.min() - 0.1 # Adjust range slightly
                max_val = cluster_centers.values.max() + 0.1
                for i in range(num_clusters):
                     fig.add_trace(go.Scatterpolar(r=cluster_centers.iloc[i][radar_features].values, theta=radar_features, fill='toself', name=f'Cluster {i}'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[min_val, max_val])), showlegend=True, title="Cluster Profiles by Strokes Gained Categories")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Players by Cluster")
                selected_cluster_name = st.selectbox('Select Cluster to View Players', cluster_centers.index)
                cluster_num_view = int(selected_cluster_name.split(' ')[1])
                cluster_players = player_stats[player_stats['cluster'] == cluster_num_view].sort_values(by='sg_total', ascending=False)
                st.write(f"Top Players in {selected_cluster_name} (by SG Total):")
                st.dataframe(cluster_players[['player', 'sg_total'] + clustering_features].head(10).style.format({'sg_total': "{:.3f}", 'sg_putt': "{:.3f}", 'sg_arg': "{:.3f}", 'sg_app': "{:.3f}", 'sg_ott': "{:.3f}"}))

                st.subheader("Cluster Interpretations")
                interpretations = []
                threshold = 0.1 # Define a threshold for strength/weakness
                for i in range(num_clusters):
                    center = cluster_centers.iloc[i]
                    strengths = [f.replace('sg_', '').upper() for f in clustering_features if center[f] > threshold]
                    weaknesses = [f.replace('sg_', '').upper() for f in clustering_features if center[f] < -threshold]
                    strength_text = f"Strong in: {', '.join(strengths)}" if strengths else "No notable strengths (above {threshold})"
                    weakness_text = f"Weak in: {', '.join(weaknesses)}" if weaknesses else "No notable weaknesses (below -{threshold})"
                    interpretations.append({"Cluster": f"Cluster {i}", "Strengths": strength_text, "Weaknesses": weakness_text})
                interp_df = pd.DataFrame(interpretations)
                st.dataframe(interp_df)

            else:
                st.warning("Not enough player data after cleaning for clustering.")
        else:
            st.error(f"Missing required columns for clustering: {', '.join(missing_cols_cluster)}")


    # --- Earnings Prediction Page ---
    elif page == "Earnings Prediction":
        st.header("Earnings Prediction Models")
        st.markdown("""
        This section uses machine learning models (Random Forest and XGBoost) to predict player earnings based on average seasonal performance metrics.
        *Note: Earnings are estimated based on finishing position and tournament purse.*
        """)

        # --- Prepare data for modeling ---
        required_cols_model = ['player', 'season', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'pos', 'purse']
        missing_cols_model = [col for col in required_cols_model if col not in df_clean.columns]

        if not missing_cols_model:
            # Create estimated earnings (simple model)
            df_model = df_clean.copy()
            df_model['estimated_earnings'] = 0.0
            # More realistic % based loosely on typical PGA distributions, extending further down
            prize_percentages = {
                1: 0.18, 2: 0.109, 3: 0.069, 4: 0.049, 5: 0.041, 10: 0.027, 20: 0.015,
                30: 0.009, 40: 0.006, 50: 0.004, 60: 0.003, 70: 0.0025
            }
            max_paid_pos = max(prize_percentages.keys())

            for i, row in df_model.iterrows():
                if pd.notna(row['pos']) and pd.notna(row['purse']) and row['purse'] > 0:
                    pos = int(row['pos'])
                    # Find the closest position key <= current position
                    perc_key = max([k for k in prize_percentages if k <= pos], default=None)
                    if perc_key is not None:
                         df_model.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[perc_key]
                    # Assign minimal percentage for positions slightly beyond the last defined key
                    elif pos <= max_paid_pos + 10:
                         df_model.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[max_paid_pos] * 0.8 # Small amount

            # Aggregate data by player for each season
            agg_metrics = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']
            player_earnings = df_model.groupby(['player', 'season']).agg(
                **{f'avg_{metric}': pd.NamedAgg(column=metric, aggfunc='mean') for metric in agg_metrics},
                total_estimated_earnings=pd.NamedAgg(column='estimated_earnings', aggfunc='sum'),
                tournament_count=pd.NamedAgg(column='tournament_name', aggfunc='nunique') # Count unique tournaments
            ).reset_index()

            # Filter out players with too few tournaments
            min_tournaments = 5
            player_earnings = player_earnings[player_earnings['tournament_count'] >= min_tournaments]
            player_earnings = player_earnings.dropna() # Drop rows with NaN in aggregated stats

            if not player_earnings.empty:
                # Select features and target
                features = [f'avg_{metric}' for metric in agg_metrics]
                X = player_earnings[features]
                y = player_earnings['total_estimated_earnings']

                # --- Split data ---
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                st.markdown("---") # Separator

                # --- Random Forest Model ---
                st.subheader("Random Forest Regression Model")
                col_rf1, col_rf2 = st.columns(2)
                with col_rf1:
                    rf_n_estimators = st.slider("RF: Number of Trees", min_value=50, max_value=500, value=100, step=50, key='rf_n')
                with col_rf2:
                    rf_max_depth = st.slider("RF: Maximum Depth", min_value=3, max_value=20, value=10, step=1, key='rf_d')

                if st.button('Train Random Forest Model', key='train_rf'):
                    with st.spinner('Training Random Forest...'):
                        rf_model = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, oob_score=True)
                        rf_model.fit(X_train, y_train)
                        y_pred_rf = rf_model.predict(X_test)
                        r2_rf = r2_score(y_test, y_pred_rf)
                        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
                        oob_rf = rf_model.oob_score_

                        st.markdown("##### Random Forest Results")
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("R² Score (Test Set)", f"{r2_rf:.4f}")
                        m_col2.metric("RMSE (Test Set)", f"${rmse_rf:,.2f}")
                        m_col3.metric("OOB Score", f"{oob_rf:.4f}") # Out-of-Bag Score

                        # Feature importance
                        rf_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
                        fig_rf_imp = px.bar(rf_feature_importance, x='Feature', y='Importance', title='RF Feature Importance in Earnings Prediction', labels={'Feature': 'Performance Metric', 'Importance': 'Importance Score'})
                        st.plotly_chart(fig_rf_imp, use_container_width=True)

                        # Prediction vs Actual
                        pred_df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
                        fig_rf_pred = px.scatter(pred_df_rf, x='Actual', y='Predicted', title='RF: Predicted vs Actual Earnings', labels={'Actual': 'Actual Earnings ($)', 'Predicted': 'Predicted Earnings ($)'}, trendline='ols')
                        fig_rf_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
                        st.plotly_chart(fig_rf_pred, use_container_width=True)


                st.markdown("---") # Separator

                # --- XGBoost Model ---
                st.subheader("XGBoost Regression Model")
                # Need to install xgboost: pip install xgboost
                col_xgb1, col_xgb2, col_xgb3 = st.columns(3)
                with col_xgb1:
                    xgb_n_estimators = st.slider("XGB: Number of Trees", min_value=50, max_value=500, value=100, step=50, key='xgb_n')
                with col_xgb2:
                    xgb_max_depth = st.slider("XGB: Maximum Depth", min_value=2, max_value=15, value=5, step=1, key='xgb_d')
                with col_xgb3:
                    xgb_learning_rate = st.slider("XGB: Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01, key='xgb_lr')

                if st.button('Train XGBoost Model', key='train_xgb'):
                    with st.spinner('Training XGBoost...'):
                        xgb_model = xgb.XGBRegressor(
                            n_estimators=xgb_n_estimators,
                            max_depth=xgb_max_depth,
                            learning_rate=xgb_learning_rate,
                            objective='reg:squarederror', # Common objective for regression
                            random_state=42,
                            n_jobs=-1 # Use all available CPU cores
                        )
                        xgb_model.fit(X_train, y_train)
                        y_pred_xgb = xgb_model.predict(X_test)
                        r2_xgb = r2_score(y_test, y_pred_xgb)
                        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

                        st.markdown("##### XGBoost Results")
                        m_col1_xgb, m_col2_xgb = st.columns(2)
                        m_col1_xgb.metric("R² Score (Test Set)", f"{r2_xgb:.4f}")
                        m_col2_xgb.metric("RMSE (Test Set)", f"${rmse_xgb:,.2f}")

                        # Feature importance
                        xgb_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_}).sort_values('Importance', ascending=False)
                        fig_xgb_imp = px.bar(xgb_feature_importance, x='Feature', y='Importance', title='XGBoost Feature Importance in Earnings Prediction', labels={'Feature': 'Performance Metric', 'Importance': 'Importance Score'})
                        st.plotly_chart(fig_xgb_imp, use_container_width=True)

                        # Prediction vs Actual
                        pred_df_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb})
                        fig_xgb_pred = px.scatter(pred_df_xgb, x='Actual', y='Predicted', title='XGBoost: Predicted vs Actual Earnings', labels={'Actual': 'Actual Earnings ($)', 'Predicted': 'Predicted Earnings ($)'}, trendline='ols')
                        fig_xgb_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
                        st.plotly_chart(fig_xgb_pred, use_container_width=True)

                st.markdown("---") # Separator

                # --- Earnings Predictor Tool ---
                st.subheader("Earnings Predictor Tool")
                st.markdown("Enter average seasonal performance metrics to predict potential earnings:")

                # Model selection for prediction
                predictor_model_choice = st.selectbox(
                    "Choose Model for Prediction",
                    ["Random Forest", "XGBoost"],
                    key='predictor_choice'
                )

                col1, col2 = st.columns(2)
                with col1:
                    sg_putt_input = st.slider("Avg SG: Putting", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key='pred_putt')
                    sg_arg_input = st.slider("Avg SG: Around the Green", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key='pred_arg')
                    sg_app_input = st.slider("Avg SG: Approach", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key='pred_app')
                with col2:
                    sg_ott_input = st.slider("Avg SG: Off the Tee", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key='pred_ott')
                    # Calculate derived metrics based on input
                    sg_t2g_input = sg_arg_input + sg_app_input + sg_ott_input
                    sg_total_input = sg_putt_input + sg_t2g_input
                    st.metric("Calculated Avg SG: Tee-to-Green", f"{sg_t2g_input:.2f}")
                    st.metric("Calculated Avg SG: Total", f"{sg_total_input:.2f}")

                if st.button('Predict Season Earnings', key='predict_earnings'):
                    with st.spinner(f'Training {predictor_model_choice} on full data & predicting...'):
                        # Train the chosen model on ALL available data for the best prediction
                        if predictor_model_choice == "Random Forest":
                            predictor_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, oob_score=True)
                        else: # XGBoost
                             predictor_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42, n_jobs=-1)

                        # Check if X and y have data before fitting
                        if not X.empty and not y.empty:
                            predictor_model.fit(X, y) # Train on full dataset X, y

                            # Create input for prediction (ensure order matches 'features')
                            input_data = np.array([[sg_putt_input, sg_arg_input, sg_app_input, sg_ott_input, sg_t2g_input, sg_total_input]])
                            input_df = pd.DataFrame(input_data, columns=features) # Use correct feature names

                            # Make prediction
                            predicted_earnings = predictor_model.predict(input_df)[0]

                            # Display prediction
                            st.success(f"Predicted Season Earnings ({predictor_model_choice}): ${predicted_earnings:,.2f}")

                            # --- Comparison and Recommendations (Based on the prediction) ---
                            st.subheader("How You Compare to Top Players")
                            top_earners = player_earnings.sort_values('total_estimated_earnings', ascending=False).head(10)

                            comparison_data = {
                                'Player': ['Your Stats'] + top_earners['player'].tolist(),
                                'Avg SG: Putt': [sg_putt_input] + top_earners['avg_sg_putt'].tolist(),
                                'Avg SG: ARG': [sg_arg_input] + top_earners['avg_sg_arg'].tolist(),
                                'Avg SG: App': [sg_app_input] + top_earners['avg_sg_app'].tolist(),
                                'Avg SG: OTT': [sg_ott_input] + top_earners['avg_sg_ott'].tolist(),
                                'Avg SG: Total': [sg_total_input] + top_earners['avg_sg_total'].tolist(),
                                'Est. Earnings': [predicted_earnings] + top_earners['total_estimated_earnings'].tolist()
                            }
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df.style.format({'Avg SG: Putt': "{:.3f}", 'Avg SG: ARG': "{:.3f}", 'Avg SG: App': "{:.3f}", 'Avg SG: OTT': "{:.3f}", 'Avg SG: Total': "{:.3f}", 'Est. Earnings': "${:,.2f}"}))


                            st.subheader("Performance Profile Comparison (Radar Chart)")
                            compare_player = st.selectbox('Select a Player to Compare With', top_earners['player'].tolist(), key='compare_player_select')
                            player_data = top_earners[top_earners['player'] == compare_player].iloc[0]

                            radar_cats = ['Avg SG: Putt', 'Avg SG: ARG', 'Avg SG: App', 'Avg SG: OTT']
                            user_values = [sg_putt_input, sg_arg_input, sg_app_input, sg_ott_input]
                            player_values = [player_data['avg_sg_putt'], player_data['avg_sg_arg'], player_data['avg_sg_app'], player_data['avg_sg_ott']]
                            min_radar = min(min(user_values), min(player_values)) - 0.2
                            max_radar = max(max(user_values), max(player_values)) + 0.2

                            fig_radar_comp = go.Figure()
                            fig_radar_comp.add_trace(go.Scatterpolar(r=user_values, theta=radar_cats, fill='toself', name='Your Stats'))
                            fig_radar_comp.add_trace(go.Scatterpolar(r=player_values, theta=radar_cats, fill='toself', name=compare_player))
                            fig_radar_comp.update_layout(polar=dict(radialaxis=dict(visible=True, range=[min_radar, max_radar])), showlegend=True, title=f"Your Performance vs {compare_player}")
                            st.plotly_chart(fig_radar_comp, use_container_width=True)

                            st.subheader("Recommendations for Improvement")
                            stats_dict = {'Putting': sg_putt_input, 'Around Green': sg_arg_input, 'Approach': sg_app_input, 'Off the Tee': sg_ott_input}
                            weakest_area = min(stats_dict, key=stats_dict.get)

                            recommendations = {
                                'Putting': ["Focus on distance control for lag putts.", "Practice short putts (3-6 feet).", "Consider green reading techniques."],
                                'Around Green': ["Improve shot selection (pitch vs. chip).", "Practice various lies (rough, bunker).", "Work on distance control for chips/pitches."],
                                'Approach': ["Focus on distance control with mid/short-irons.", "Practice shots from key yardages (e.g., 100-175 yards).", "Work on controlling trajectory and spin."],
                                'Off the Tee': ["Prioritize accuracy (fairways hit).", "Develop a reliable 'fairway finder' shot.", "Optimize launch conditions (driver fitting)."]
                            }
                            st.write(f"Based on your profile, your biggest relative weakness appears to be: **{weakest_area}**")
                            st.write("Consider focusing on:")
                            for rec in recommendations[weakest_area]:
                                st.write(f"- {rec}")

                            # Potential earnings improvement simulation
                            improvement_value = 0.3 # Simulate a modest 0.3 SG improvement
                            improved_values = list(stats_dict.values())
                            weakest_index = list(stats_dict.keys()).index(weakest_area)
                            improved_values[weakest_index] += improvement_value

                            # Recalculate T2G and Total
                            improved_t2g = improved_values[1] + improved_values[2] + improved_values[3]
                            improved_total = improved_values[0] + improved_t2g

                            improved_input_data = np.array([[improved_values[0], improved_values[1], improved_values[2], improved_values[3], improved_t2g, improved_total]])
                            improved_input_df = pd.DataFrame(improved_input_data, columns=features)
                            improved_earnings = predictor_model.predict(improved_input_df)[0]
                            earnings_increase = improved_earnings - predicted_earnings

                            st.write(f"If you improve your {weakest_area} by {improvement_value:.1f} strokes gained per round:")
                            st.metric(label="Potential New Predicted Earnings", value=f"${improved_earnings:,.2f}", delta=f"${earnings_increase:,.2f} (+{(earnings_increase/predicted_earnings)*100:.1f}%)")

                        else:
                             st.warning("Not enough data available to train the prediction model.")


            else:
                st.warning("Not enough player data after aggregation and filtering to build prediction models.")
        else:
            st.error(f"Missing required columns for modeling: {', '.join(missing_cols_model)}. Cannot proceed with Earnings Prediction.")

# --- Handle case where data loading failed ---
elif df is None:
    # Error message is already displayed by load_data()
    st.stop() # Stop the script execution if data isn't loaded