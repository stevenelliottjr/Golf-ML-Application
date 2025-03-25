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
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="PGA Tour Performance Analytics",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("PGA Tour Performance Analytics")
st.markdown("""
This application analyzes professional golf performance metrics and their relationship to earnings.
Explore different visualizations, player clustering, and predictive models to understand key factors
influencing golfer performance and financial success.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Data Overview", "Exploratory Analysis", "Performance Clustering", "Earnings Prediction"]
)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('golf_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Clean the data
    def clean_data(df):
        # Make a copy of the dataframe
        df_clean = df.copy()
        
        # Drop rows with missing values in key columns
        key_columns = ['sg_total', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g']
        df_clean = df_clean.dropna(subset=key_columns)
        
        # Convert finish position to numeric
        # Remove any non-numeric characters (like 'T' for tied positions)
        df_clean['pos'] = pd.to_numeric(df_clean['pos'], errors='coerce')
        
        # Convert date to datetime
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        # Drop rows where date conversion failed
        df_clean = df_clean.dropna(subset=['date'])
        
        # Fill remaining NaN values with the median for numerical columns
        numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        return df_clean
    
    # Clean the data
    df_clean = clean_data(df)
    
    # Data Overview Page
    if page == "Data Overview":
        st.header("Dataset Overview")
        
        # Show basic statistics
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df_clean.head())
        
        # Column descriptions
        st.subheader("Column Descriptions")
        column_descriptions = {
            'Player_initial_last': 'Player name (initial + last name)',
            'tournament id': 'Unique identifier for the tournament',
            'player id': 'Unique identifier for the player',
            'hole_par': 'Par value for the hole',
            'strokes': 'Number of strokes taken',
            'made_cut': 'Whether the player made the cut (1) or not (0)',
            'pos': 'Final position in the tournament',
            'player': 'Full player name',
            'tournament_name': 'Name of the tournament',
            'course': 'Course name',
            'date': 'Tournament date',
            'purse': 'Tournament prize money (in USD)',
            'season': 'Season year',
            'no_cut': 'Whether the tournament has no cut (1) or has a cut (0)',
            'Finish': 'Finish position as string (including ties)',
            'sg_putt': 'Strokes gained: putting',
            'sg_arg': 'Strokes gained: around the green',
            'sg_app': 'Strokes gained: approach',
            'sg_ott': 'Strokes gained: off the tee',
            'sg_t2g': 'Strokes gained: tee to green',
            'sg_total': 'Strokes gained: total'
        }
        
        for col, desc in column_descriptions.items():
            if col in df_clean.columns:
                st.write(f"**{col}**: {desc}")
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing_vals = df_clean.isnull().sum()
        missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)
        if len(missing_vals) > 0:
            st.write("Columns with missing values:")
            st.write(missing_vals)
            
            # Plot missing values
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_vals.plot(kind='bar', ax=ax)
            plt.title('Missing Values by Column')
            plt.ylabel('Count')
            plt.xlabel('Columns')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No missing values in the cleaned dataset.")
    
    # Exploratory Analysis Page
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        # Strokes Gained Analysis
        st.subheader("Strokes Gained Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of SG Total
            fig = px.histogram(df_clean, x='sg_total', 
                               title='Distribution of Strokes Gained: Total',
                               labels={'sg_total': 'Strokes Gained: Total'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strokes gained components comparison
            sg_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott']
            sg_data = df_clean[sg_cols].mean().reset_index()
            sg_data.columns = ['Category', 'Average Strokes Gained']
            
            fig = px.bar(sg_data, x='Category', y='Average Strokes Gained',
                         title='Average Strokes Gained by Category',
                         labels={'Category': 'Strokes Gained Category', 
                                'Average Strokes Gained': 'Average Value'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'pos', 'purse']
        corr_df = df_clean[numeric_cols].corr()
        
        fig = px.imshow(corr_df, 
                       title='Correlation Matrix of Performance Metrics',
                       labels=dict(color="Correlation"),
                       x=corr_df.columns,
                       y=corr_df.columns,
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Position vs. Strokes Gained
        st.subheader("Position vs. Strokes Gained")
        
        sg_position_option = st.selectbox(
            'Select Strokes Gained Category',
            ['sg_total', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g']
        )
        
        # Filter out extreme outliers for better visualization (positions > 100)
        filtered_df = df_clean[df_clean['pos'] <= 100]
        
        fig = px.scatter(filtered_df, x=sg_position_option, y='pos',
                        title=f'Tournament Position vs. {sg_position_option}',
                        labels={sg_position_option: sg_position_option.replace('_', ' ').title(), 
                               'pos': 'Tournament Position'},
                        trendline='ols')
        
        # Invert y-axis since lower position is better
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        
        # Verify required columns exist in the cleaned DataFrame
        required_columns = ['sg_total', 'tournament_name', 'purse', 'player', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]

        if missing_columns:
            st.error(f"The following required columns are missing in the dataset: {', '.join(missing_columns)}")
        else:
            # Tournament Purse Analysis
            st.subheader("Tournament Purse Analysis")
            
            # Highest purse tournaments
            top_tournaments = df_clean.groupby('tournament_name')['purse'].mean().sort_values(ascending=False).head(10)
            
            fig = px.bar(top_tournaments, 
                        title='Top 10 Tournaments by Prize Money',
                        labels={'value': 'Average Purse (USD)', 'tournament_name': 'Tournament'})
            st.plotly_chart(fig, use_container_width=True)

    # Performance Clustering Page
    elif page == "Performance Clustering":
        st.header("Player Performance Clustering")
        
        st.markdown("""
        This analysis uses K-means clustering to group golfers based on their performance patterns.
        The algorithm identifies natural groupings in the data that can reveal different playing styles and strengths.
        """)
        
        # Prepare data for clustering
        # Aggregate data by player
        player_stats = df_clean.groupby('player')[['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']].mean().reset_index()
        
        # Ensure 'sg_total' is included in the player_stats DataFrame
        if 'sg_total' not in player_stats.columns:
            st.error("The column 'sg_total' is missing in the aggregated player statistics. Please check the data.")
        else:
            # Drop rows with NaN values
            player_stats = player_stats.dropna()
        
            # Select features for clustering
            features = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott']
            X = player_stats[features]
        
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
            # Determine optimal number of clusters
            st.subheader("Optimal Number of Clusters")
        
            # Elbow method calculation
            inertia = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
        
            # Plot Elbow Method
            fig = px.line(x=k_range, y=inertia, markers=True,
                         title='Elbow Method for Optimal k',
                         labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'})
            st.plotly_chart(fig, use_container_width=True)
        
            # Let user select number of clusters
            num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)
        
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            player_stats['cluster'] = kmeans.fit_predict(X_scaled)
        
            # Display cluster centers
            st.subheader("Cluster Centers")
        
            # Transform cluster centers back to original scale
            cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                          columns=features)
            cluster_centers.index.name = 'Cluster'
            cluster_centers.index = ["Cluster " + str(i) for i in range(num_clusters)]
        
            st.write(cluster_centers)
        
            # Visualize clusters
            st.subheader("Cluster Visualization")
        
            # Let user select features for visualization
            x_axis = st.selectbox('Select X-axis feature', features, index=2)  # Default to sg_app
            y_axis = st.selectbox('Select Y-axis feature', features, index=0)  # Default to sg_putt
        
            # Create Scatter plot
            fig = px.scatter(player_stats, x=x_axis, y=y_axis, color='cluster',
                            title=f'Player Clustering by {x_axis} and {y_axis}',
                            labels={x_axis: x_axis.replace('_', ' ').title(), 
                                   y_axis: y_axis.replace('_', ' ').title()},
                            hover_data=['player'])
            st.plotly_chart(fig, use_container_width=True)
        
            # Display radar chart for cluster profiles
            st.subheader("Cluster Profiles (Radar Chart)")
        
            # Create radar chart
            fig = go.Figure()
        
            for i in range(num_clusters):
                fig.add_trace(go.Scatterpolar(
                    r=cluster_centers.iloc[i].values,
                    theta=features,
                    fill='toself',
                    name=f'Cluster {i}'
                ))
        
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[cluster_centers.values.min()-0.2, cluster_centers.values.max()+0.2]
                    )),
                showlegend=True,
                title="Cluster Profiles by Strokes Gained Categories"
            )
        
            st.plotly_chart(fig, use_container_width=True)
        
            # Show players in each cluster
            st.subheader("Players by Cluster")
        
            selected_cluster = st.selectbox(
                'Select Cluster to View Players',
                [f"Cluster {i}" for i in range(num_clusters)]
            )
        
            cluster_num = int(selected_cluster.split(' ')[1])
            cluster_players = player_stats[player_stats['cluster'] == cluster_num]
        
            if 'sg_total' in cluster_players.columns:
                cluster_players = cluster_players.sort_values(by='sg_total', ascending=False)
                st.write(f"Top Players in {selected_cluster}:")
                st.dataframe(cluster_players.head(10))
            else:
                st.error("The column 'sg_total' is missing in the cluster data.")
        
            # Cluster interpretations
            st.subheader("Cluster Interpretations")
        
            # Interpret clusters based on their centers
            interpretations = []
            for i in range(num_clusters):
                center = cluster_centers.iloc[i]
                strengths = []
                weaknesses = []
                
                for feature in features:
                    if center[feature] > 0.2:
                        strengths.append(feature.replace('sg_', '').upper())
                    elif center[feature] < -0.2:
                        weaknesses.append(feature.replace('sg_', '').upper())
                        
                if len(strengths) > 0:
                    strength_text = "Strong in: " + ", ".join(strengths)
                else:
                    strength_text = "No notable strengths"
                    
                if len(weaknesses) > 0:
                    weakness_text = "Weak in: " + ", ".join(weaknesses)
                else:
                    weakness_text = "No notable weaknesses"
                    
                interpretations.append({
                    "cluster": f"Cluster {i}",
                    "strengths": strength_text,
                    "weaknesses": weakness_text
                })
                
            interp_df = pd.DataFrame(interpretations)
            st.write(interp_df)
    
    # Earnings Prediction Page
    elif page == "Earnings Prediction":
        st.header("Earnings Prediction Models")
        
        st.markdown("""
        This section uses machine learning models to predict player earnings based on performance metrics.
        Explore how different statistical models evaluate the relationship between golf performance and financial success.
        """)
        
        # Prepare data for modeling
        # Create a proxy for earnings based on tournament position and purse
        df_clean['estimated_earnings'] = 0.0
        
        # Simple model: Winner gets 18% of purse, 2nd gets 10.8%, etc.
        # Basic prize distribution approximation
        prize_percentages = {
            1: 0.18,  # Winner: 18% of purse
            2: 0.108, # 2nd: 10.8%
            3: 0.068, # 3rd: 6.8%
            4: 0.048, # 4th: 4.8%
            5: 0.04,  # 5th: 4.0%
            10: 0.023, # 10th: 2.3%
            20: 0.014, # 20th: 1.4%
            30: 0.009, # 30th: 0.9%
            40: 0.007, # 40th: 0.7%
            50: 0.006, # 50th: 0.6%
            60: 0.0056, # 60th: 0.56%
            70: 0.0052, # 70th: 0.52%
        }
        
        # Apply a simple formula based on position
        for i, row in df_clean.iterrows():
            if pd.isna(row['pos']) or pd.isna(row['purse']):
                continue
                
            pos = int(row['pos'])
            if pos == 1:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[1]
            elif pos == 2:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[2]
            elif pos == 3:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[3]
            elif pos == 4:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[4]
            elif pos == 5:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[5]
            elif 6 <= pos <= 10:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[10]
            elif 11 <= pos <= 20:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[20]
            elif 21 <= pos <= 30:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[30]
            elif 31 <= pos <= 40:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[40]
            elif 41 <= pos <= 50:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[50]
            elif 51 <= pos <= 60:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[60]
            elif pos > 60:
                df_clean.at[i, 'estimated_earnings'] = row['purse'] * prize_percentages[70]
        
        # Aggregate data by player for each season
        player_earnings = df_clean.groupby(['player', 'season'])[['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'estimated_earnings']].agg({
            'sg_putt': 'mean',
            'sg_arg': 'mean',
            'sg_app': 'mean',
            'sg_ott': 'mean',
            'sg_t2g': 'mean',
            'sg_total': 'mean',
            'estimated_earnings': 'sum'
        }).reset_index()
        
        # Filter out players with too few tournaments
        player_counts = df_clean.groupby(['player', 'season']).size().reset_index(name='tournament_count')
        player_earnings = player_earnings.merge(player_counts, on=['player', 'season'])
        player_earnings = player_earnings[player_earnings['tournament_count'] >= 5]
        
        # Drop rows with NaN values
        player_earnings = player_earnings.dropna()
        
        # Select features for modeling
        features = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']
        X = player_earnings[features]
        y = player_earnings['estimated_earnings']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest Model
        st.subheader("Random Forest Regression Model")
        
        # Hyperparameters
        n_estimators = st.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50)
        max_depth = st.slider("Maximum Depth", min_value=3, max_value=20, value=10, step=1)
        
        # Train model
        if st.button('Train Random Forest Model'):
            with st.spinner('Training model...'):
                # Create and train the model
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate model
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Display metrics
                col1, col2 = st.columns(2)
                col1.metric("R² Score", f"{r2:.4f}")
                col2.metric("RMSE", f"${rmse:,.2f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance, x='Feature', y='Importance',
                            title='Feature Importance in Earnings Prediction',
                            labels={'Feature': 'Performance Metric', 'Importance': 'Importance Score'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction vs Actual
                st.subheader("Predicted vs Actual Earnings")
                prediction_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                }).reset_index(drop=True)
                
                fig = px.scatter(prediction_df, x='Actual', y='Predicted',
                                title='Predicted vs Actual Earnings',
                                labels={'Actual': 'Actual Earnings ($)', 'Predicted': 'Predicted Earnings ($)'},
                                trendline='ols')
                
                # Add 45-degree line
                fig.add_trace(
                    go.Scatter(
                        x=[prediction_df['Actual'].min(), prediction_df['Actual'].max()],
                        y=[prediction_df['Actual'].min(), prediction_df['Actual'].max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Earnings predictor tool
        st.subheader("Earnings Predictor Tool")
        st.markdown("Enter performance metrics to predict potential earnings:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sg_putt_input = st.slider("Strokes Gained: Putting", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
            sg_arg_input = st.slider("Strokes Gained: Around the Green", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        
        with col2:
            sg_app_input = st.slider("Strokes Gained: Approach", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
            sg_ott_input = st.slider("Strokes Gained: Off the Tee", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        
        # Calculate derived metrics
        sg_t2g_input = sg_arg_input + sg_app_input + sg_ott_input
        sg_total_input = sg_putt_input + sg_t2g_input
        
        # Display derived metrics
        st.write(f"Strokes Gained Tee-to-Green: {sg_t2g_input:.2f}")
        st.write(f"Strokes Gained Total: {sg_total_input:.2f}")
        
        # Create prediction button
        if st.button('Predict Season Earnings'):
            # Train a model on all data (for prediction tool)
            predictor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            predictor.fit(X, y)
            
            # Create input for prediction
            input_data = np.array([[sg_putt_input, sg_arg_input, sg_app_input, sg_ott_input, sg_t2g_input, sg_total_input]])
            
            # Make prediction
            predicted_earnings = predictor.predict(input_data)[0]
            
            # Display prediction
            st.success(f"Predicted Season Earnings: ${predicted_earnings:,.2f}")
            
            # Compare to top players
            st.subheader("How You Compare to Top Players")
            
            # Get top 10 earners
            top_earners = player_earnings.sort_values('estimated_earnings', ascending=False).head(10)
            
            # Format for comparison
            comparison_df = pd.DataFrame({
                'Player': ['Your Stats'] + top_earners['player'].tolist(),
                'SG: Putting': [sg_putt_input] + top_earners['sg_putt'].tolist(),
                'SG: Around Green': [sg_arg_input] + top_earners['sg_arg'].tolist(),
                'SG: Approach': [sg_app_input] + top_earners['sg_app'].tolist(), 
                'SG: Off the Tee': [sg_ott_input] + top_earners['sg_ott'].tolist(),
                'SG: Total': [sg_total_input] + top_earners['sg_total'].tolist(),
                'Est. Earnings': [predicted_earnings] + top_earners['estimated_earnings'].tolist()
            })
            
            # Format earnings column
            comparison_df['Est. Earnings'] = comparison_df['Est. Earnings'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(comparison_df)
            
            # Radar chart comparison
            st.subheader("Performance Profile Comparison")
            
            # Let user select a player to compare with
            compare_player = st.selectbox(
                'Select a Player to Compare With',
                top_earners['player'].tolist()
            )
            
            # Get player data
            player_data = top_earners[top_earners['player'] == compare_player].iloc[0]
            
            # Create radar chart
            categories = ['SG: Putting', 'SG: Around Green', 'SG: Approach', 'SG: Off the Tee']
            user_values = [sg_putt_input, sg_arg_input, sg_app_input, sg_ott_input]
            player_values = [player_data['sg_putt'], player_data['sg_arg'], player_data['sg_app'], player_data['sg_ott']]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=user_values,
                theta=categories,
                fill='toself',
                name='Your Stats'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=player_values,
                theta=categories,
                fill='toself',
                name=compare_player
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[min(min(user_values), min(player_values))-0.5, 
                              max(max(user_values), max(player_values))+0.5]
                    )),
                showlegend=True,
                title=f"Your Performance vs {compare_player}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Improvement recommendations
            st.subheader("Recommendations for Improvement")
            
            # Find weakest area
            categories_short = ['Putting', 'Around Green', 'Approach', 'Off the Tee']
            values = [sg_putt_input, sg_arg_input, sg_app_input, sg_ott_input]
            weakest_index = values.index(min(values))
            weakest_area = categories_short[weakest_index]
            
            # Generate recommendations
            recommendations = {
                'Putting': [
                    "Focus on distance control for lag putts to reduce three-putts",
                    "Practice short putts (3-5 feet) to improve conversion rate",
                    "Work with a putting coach to refine your stroke mechanics"
                ],
                'Around Green': [
                    "Improve shot selection around the green (pitch vs. chip vs. putt)",
                    "Practice different lie conditions in the rough and bunkers",
                    "Work on distance control for chip shots"
                ],
                'Approach': [
                    "Focus on improving distance control with mid-irons",
                    "Practice shots from 125-175 yards to increase GIR percentage",
                    "Work on controlling ball flight and trajectory in different conditions"
                ],
                'Off the Tee': [
                    "Focus on driving accuracy rather than just distance",
                    "Develop a reliable 'fairway finder' shot for tight holes",
                    "Work with a coach on optimizing launch conditions for your swing speed"
                ]
            }
            
            st.write(f"Based on your profile, your main area for improvement is: **{weakest_area}**")
            
            st.write("Recommendations:")
            for rec in recommendations[weakest_area]:
                st.write(f"- {rec}")
            
            # Potential earnings improvement
            improvement_value = 0.5  # Assume 0.5 strokes gained improvement
            
            # Create improved stats
            improved_values = values.copy()
            improved_values[weakest_index] += improvement_value
            
            # Calculate new total
            improved_total = sum(improved_values)
            
            # Create input for prediction
            improved_input = np.array([[
                improved_values[0], 
                improved_values[1], 
                improved_values[2], 
                improved_values[3],
                improved_values[1] + improved_values[2] + improved_values[3],  # t2g
                improved_total
            ]])
            
            # Make prediction
            improved_earnings = predictor.predict(improved_input)[0]
            
            # Display potential improvement
            earnings_increase = improved_earnings - predicted_earnings
            
            st.write(f"If you improve your {weakest_area} by 0.5 strokes gained:")
            st.write(f"- New predicted earnings: ${improved_earnings:,.2f}")
            st.write(f"- Earnings increase: ${earnings_increase:,.2f} (+{(earnings_increase/predicted_earnings)*100:.1f}%)")

# Run the app
if df is None:
    st.error("Failed to load data. Please check the data file path.")