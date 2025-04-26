import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import itertools



st.title("S&P 500 Forecast")

df = pd.read_excel(r"C:\Users\aramc\Desktop\Final Year Project - 2025\sp500_data.xlsx", sheet_name="Clean", parse_dates=["Date"])
mets = pd.read_excel(r"C:\Users\aramc\Desktop\Final Year Project - 2025\predictions and metrics.xlsx", sheet_name="Model Metrics")
fpreds = pd.read_excel(r"C:\Users\aramc\Desktop\Final Year Project - 2025\predictions and metrics.xlsx", sheet_name="Future Predictions")


df2 = df.copy()
df2["Day_of_Week"] = df2["Date"].dt.day_name()
df2["Month"] = df2["Date"].dt.month
df2["Year"] = df2["Date"].dt.year

df['Date'] = pd.to_datetime(df['Date'])
df["Year"] = df['Date'].dt.year

lrdf = df.copy()
rfdf = df.copy()
madf = df.copy()
ardf = df.copy()

#year_options = ["All Years"] + sorted(df['Year'].unique().tolist())  
#selected_year = st.selectbox("Select Year", year_options)


# Trendlines for Open, High, Low, Closed based on Year Selected
# Filter the dataframe based on selected year
#if selected_year == "All Years":
#    df1 = df
#else:
#    df1 = df[df['Year'] == selected_year]


tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 EDA", "📊 EDA", "🔗 Relationships", "🤖 Models", "🔎 Metrics and Predictions"])

with tab1:

    year_options = ["All Years"] + sorted(df['Year'].unique().tolist())  
    selected_year = st.selectbox("Select Year", year_options)

    if selected_year == "All Years":
        df1 = df
    else:
        df1 = df[df['Year'] == selected_year]

    ## Trendlines for Open, High , Low,Close and Volume
    df1['Date'] = pd.to_datetime(df1['Date'])  

    #subplots for each plot using plotly
    fig = make_subplots(
        rows=3, 
        cols=2, 
        subplot_titles=["S&P 500 Open Price", "S&P 500 High Price", 
                        "S&P 500 Low Price", "S&P 500 Close Price", 
                        "S&P 500 Volume", ""], 
        shared_yaxes=True  # Shared y-axis for consistency
    )

    # Open Price plot
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Open'], mode='lines', name='Open'), row=1, col=1)

    # High Price plot
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['High'], mode='lines', name='High'), row=1, col=2)

    # Low Price plot
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Low'], mode='lines', name='Low'), row=2, col=1)

    # Close Price plot
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Close'], mode='lines', name='Close'), row=2, col=2)

    # Volume plot
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Volume'], mode='lines', name='Volume'), row=3, col=1)

    #Hide the last subplot axis (the empty one)
    fig.update_xaxes(showticklabels=False, row=3, col=2)
    fig.update_yaxes(showticklabels=False, row=3, col=2)

    #Update layout for better appearance
    fig.update_layout(
        title_text="S&P 500 Data", 
        height=900, 
        showlegend=True,
    )

    #Show the interactive plot in Streamlit
    st.plotly_chart(fig)

with tab2:
    st.write("Exploratory Data Analysis")
    # Ensure Date column is in datetime format
    df2['Date'] = pd.to_datetime(df2['Date'])

    # Define proper ordering for days of the week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Map month numbers to names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    # User selects grouping method
    grouping = st.selectbox("Group Data By:", ["Day of Week", "Month", "Year"])

    # Aggregate data based on user selection
    if grouping == "Day of Week":
        grouped_df = df2.groupby("Day_of_Week")[["Open", "High", "Low", "Close", "Volume"]].mean().reset_index()
        grouped_df["Day_of_Week"] = pd.Categorical(grouped_df["Day_of_Week"], categories=day_order, ordered=True)
        grouped_df = grouped_df.sort_values("Day_of_Week")  
        x_col = "Day_of_Week"

    elif grouping == "Month":
        grouped_df = df2.groupby("Month")[["Open", "High", "Low", "Close", "Volume"]].mean().reset_index()
        grouped_df["Month"] = grouped_df["Month"].map(month_names) 
        grouped_df["Month"] = pd.Categorical(grouped_df["Month"], categories=list(month_names.values()), ordered=True)
        grouped_df = grouped_df.sort_values("Month") 
        x_col = "Month"

    else: 
        grouped_df = df2.groupby("Year")[["Open", "High", "Low", "Close", "Volume"]].mean().reset_index()
        x_col = "Year"

    #bar charts for each variable
    st.subheader(f"Average Values by {grouping}")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        fig = px.bar(
            grouped_df,
            x=x_col,  
            y=col,
            title=f"Average {col} by {grouping}",
            labels={x_col: grouping, col: f"Avg {col}"},
            color=col,
        )
        st.plotly_chart(fig)


with tab3:
    variables = ["Open", "High", "Low", "Close", "Volume"]

    # --------------------------------
    # 📊 Interactive Scatterplots
    # --------------------------------
    st.header("📉 Scatterplots of Variable Pairs")

    #unique variable pairs
    pairs = list(itertools.combinations(variables, 2))

    #Dropdown to select scatterplot pair
    selected_pair = st.selectbox("Select Variable Pair", [f"{x} vs {y}" for x, y in pairs])

    #selected variables
    x_var, y_var = selected_pair.split(" vs ")

    #interactive scatterplot
    fig_scatter = px.scatter(df, x=x_var, y=y_var, title=f"{x_var} vs {y_var}", 
                            trendline="ols", opacity=0.7, color_discrete_sequence=["blue"])
    st.plotly_chart(fig_scatter)

    st.header("🔢 Correlation Matrix")
    correlation_matrix = df[variables].corr()

    #heatmap 
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale="RdBu",  
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    #text annotations for better readability
    for i in range(len(variables)):
        for j in range(len(variables)):
            fig_corr.add_annotation(
                x=correlation_matrix.columns[j],
                y=correlation_matrix.index[i],
                text=f"{correlation_matrix.iloc[i, j]:.5f}",
                showarrow=False,
                font=dict(color="black", size=12, family="Arial Black"),
            )

    # Update layout for better grid visibility
    fig_corr.update_layout(
        title="Correlation Matrix for the Numerical Variables",
        xaxis=dict(showgrid=True, zeroline=False),  # Add gridlines
        yaxis=dict(showgrid=True, zeroline=False),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Display in Streamlit
    st.plotly_chart(fig_corr)

with tab4:
    model = st.selectbox("Select Model to View:", ["Linear Regression Model", "Random Forest Model","3-Day Moving Average", "7-Day Moving Average","14-Day Moving Average", "ARIMA"])

    if model == "Linear Regression Model":
        st.write("LINEAR REGRESSION MODEL")
        lr = lrdf[["Date", "Open", "High", "Low", "Close", "Volume"]]
        lr['Candle_Body'] = abs(lr['Open'] - lr['Close'])
        lr['Next_day_Open']=lr['Open'].shift(-1)
        lr['Next_day_Close']=lr['Close'].shift(-1)
        lr = lr.dropna()

        #Excluding 'Date' and 'Next_day_Close' from scaling
        columns_to_scale = [col for col in lr.columns if col not in ["Date", "Next_day_Close"]]
        scaler = MinMaxScaler()
        lr[columns_to_scale] = scaler.fit_transform(lr[columns_to_scale])

        lr_X = lr.drop(columns=["Date", "Next_day_Close"])
        lr_y = lr["Next_day_Close"]

        lr["Date"] = pd.to_datetime(lr["Date"])
        split_date = "2024-01-01"

        #Training set (2020-2023)
        lr_X_train = lr_X[lr["Date"] < split_date]
        lr_y_train = lr_y[lr["Date"] < split_date]

        #Testing set (2024)
        lr_X_test = lr_X[lr["Date"] >= split_date]
        lr_y_test = lr_y[lr["Date"] >= split_date]

        #Initializing and Training the Model
        lr_model = LinearRegression()
        #model.fit(X_train, y_train.ravel())
        lr_model.fit(lr_X_train, lr_y_train)

        #predictions
        lr_y_pred = lr_model.predict(lr_X_test)

        #Model Evaluation
        lr_r2_test = metrics.r2_score(lr_y_test, lr_y_pred)
        lr_r2_train = lr_model.score(lr_X_train, lr_y_train)
        lr_mse = metrics.mean_squared_error(lr_y_test, lr_y_pred)
        lr_rmse = np.sqrt(lr_mse)
        lr_mae = metrics.mean_absolute_error(lr_y_test, lr_y_pred)

        lr_results = pd.DataFrame({
        "Date": lr["Date"][lr["Date"] >= split_date], 
        "Actual": lr_y_test.values, 
        "Predicted": lr_y_pred
        })

        # Plot
        lr_fig = go.Figure()

        # Training Data
        lr_fig.add_trace(go.Scatter(
            x=lr["Date"][lr["Date"] < split_date], 
            y=lr_y_train, 
            mode='lines', 
            name='Train Data',
            line=dict(color='blue')
        ))

        # Testing Data (Actual)
        lr_fig.add_trace(go.Scatter(
            x=lr_results["Date"], 
            y=lr_results["Actual"], 
            mode='lines', 
            name='Test Data (Actual)',
            line=dict(color='green')
        ))

        # Predictions
        lr_fig.add_trace(go.Scatter(
            x=lr_results["Date"], 
            y=lr_results["Predicted"], 
            mode='lines', 
            name='Predictions',
            line=dict(color='red', dash='dot')
        ))

        # Customize layout
        lr_fig.update_layout(
            title="Linear Regression Model Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Next Day Close Price",
            legend_title="Legend",
            template="plotly_dark"
        )

        # Display plot in Streamlit
        st.plotly_chart(lr_fig)

        st.write(f'**Mean Absolute Error (MAE):** {lr_mae:.4f}')
        st.write(f'**Mean Squared Error (MSE):** {lr_mse:.4f}')
        st.write(f'**Root Mean Squared Error (RMSE):** {lr_rmse:.4f}')
    
    elif model == "Random Forest Model":
        st.write("RANDOM FOREST MODEL")

        rfdata = rfdf[["Date", "Open", "High", "Low", "Close", "Volume"]]
        rfdata['Candle_Body'] = abs(rfdata['Open'] - rfdata['Close'])
        rfdata['Next_day_Open']=rfdata['Open'].shift(-1)
        rfdata['Next_day_Close']=rfdata['Close'].shift(-1)
        rfdata = rfdata.dropna()

        #Excluding 'Date' and 'Next_day_Close' from scaling
        cols_to_scale = [col for col in rfdata.columns if col not in ["Date", "Next_day_Close"]]
        scaler = MinMaxScaler()
        rfdata[cols_to_scale] = scaler.fit_transform(rfdata[cols_to_scale])

        rf_X = rfdata.drop(columns=["Date", "Next_day_Close"])
        rf_y = rfdata["Next_day_Close"]

        #rfdata["Date"] = pd.to_datetime(["Date"])
        split_date = "2024-01-01"

        #Training set (2020-2023)
        rf_X_train = rf_X[rfdata["Date"] < split_date]
        rf_y_train = rf_y[rfdata["Date"] < split_date]

        #Testing set (2024)
        rf_X_test = rf_X[rfdata["Date"] >= split_date]
        rf_y_test = rf_y[rfdata["Date"] >= split_date]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit the model to the training data
        rf_model.fit(rf_X_train, rf_y_train)

        # Make predictions on the test data
        y_pred_rf = rf_model.predict(rf_X_test)

        # Convert results to DataFrame for plotting
        rf_results = pd.DataFrame({
            "Date": rfdata["Date"][rfdata["Date"] >= split_date],
            "Actual": rf_y_test.values,
            "Predicted": y_pred_rf
        })

        # Plot
        fig = go.Figure()

        # Training Data
        fig.add_trace(go.Scatter(
            x=rfdata["Date"][rfdata["Date"] < split_date],
            y=rf_y_train,
            mode='lines',
            name='Train Data',
            line=dict(color='blue')
        ))

        # Test Data (Actual)
        fig.add_trace(go.Scatter(
            x=rf_results["Date"],
            y=rf_results["Actual"],
            mode='lines',
            name='Test Data (Actual)',
            line=dict(color='green')
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=rf_results["Date"],
            y=rf_results["Predicted"],
            mode='lines',
            name='Predictions',
            line=dict(color='red', dash='dot')
        ))

        # Customize layout
        fig.update_layout(
            title="Stock Price Prediction (Random Forest)",
            xaxis_title="Date",
            yaxis_title="Next Day Close Price",
            legend_title="Legend",
            template="plotly_dark"
        )

        # Display plot in Streamlit
        st.plotly_chart(fig)

        # Evaluate the model using R², MSE, and RMSE
        mse_rf = mean_squared_error(rf_y_test, y_pred_rf)
        mae_rf = metrics.mean_absolute_error(rf_y_test, y_pred_rf)
        rmse_rf = mse_rf ** 0.5

        # Display results
        st.write(f'**Mean Absolute Error (MAE):** {mae_rf:.4f}')
        st.write(f'**Mean Squared Error (MSE):** {mse_rf:.4f}')
        st.write(f'**Root Mean Squared Error (RMSE):** {rmse_rf:.4f}')

    elif model == "3-Day Moving Average":
        st.write("3-DAY MOVING AVERAGE")
        madf["Date"] = pd.to_datetime(madf["Date"])
        madf = madf.sort_values(by="Date")

        # Moving average features
        madf["Close_3MA"] = madf["Close"].rolling(window=3).mean()
        madf["Close_7MA"] = madf["Close"].rolling(window=7).mean()
        madf["Close_14MA"] = madf["Close"].rolling(window=14).mean()
        madf.dropna(inplace=True)

        # Features and target
        ma_X = madf[["Close_3MA", "Close_7MA", "Close_14MA"]]
        ma_Y = madf["Close"]
        dates = madf["Date"]
        split_date = "2024-01-01"

        # Train-test split
        ma_X_train = ma_X[madf["Date"] < split_date]
        ma_Y_train = ma_Y[madf["Date"] < split_date]
        ma_X_test = ma_X[madf["Date"] >= split_date]
        ma_Y_test = ma_Y[madf["Date"] >= split_date]
        dates_test = dates[madf["Date"] >= split_date]

        # Use the last known Close_3MA as predictions
        ma_Y_pred = ma_X_test["Close_3MA"]

        # Future Predictions (Next 10 Days)
        future_dates = pd.date_range(start=dates_test.max(), periods=11, freq="D")[1:]

        # Initialize future predictions list with the last known values
        future_closes = list(ma_Y_test[-3:].values)  # Use last 3 actual closing values

        # Generate future predictions iteratively
        for _ in range(len(future_dates)):
            # Calculate new moving average based on the last 3 values
            future_3MA = np.mean(future_closes[-3:])
            
            # Append new predicted close price
            future_closes.append(future_3MA)

        # Remove initial test values, keeping only new future predictions
        future_preds = future_closes[3:]

        # Create DataFrame for Future Predictions
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": future_preds
        })

        # Evaluate Model
        three_mae = metrics.mean_absolute_error(ma_Y_test, ma_Y_pred)
        three_mse = metrics.mean_squared_error(ma_Y_test, ma_Y_pred)
        three_rmse = np.sqrt(three_mse)

        # Interactive Plotly Chart
        ma_fig = go.Figure()

        # Train Data
        ma_fig.add_trace(go.Scatter(
            x=dates[madf["Date"] < split_date], 
            y=ma_Y_train, 
            mode='lines', 
            name='Train Data',
            line=dict(color='blue')
        ))

        # Test Data (Actual)
        ma_fig.add_trace(go.Scatter(
            x=dates_test, 
            y=ma_Y_test, 
            mode='lines', 
            name='Test Data (Actual)',
            line=dict(color='green')
        ))

        # Test Data Predictions
        ma_fig.add_trace(go.Scatter(
            x=dates_test, 
            y=ma_Y_pred, 
            mode='lines', 
            name='Predictions',
            line=dict(color='red', dash='dot')
        ))

        # Future Predictions
        ma_fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_preds, 
            mode='lines', 
            name='Future Predictions (Next 10 Days)',
            line=dict(color='orange', dash='dash')
        ))

        # Customize layout
        ma_fig.update_layout(
            title="Three-Day Moving Average Model Predictions",
            xaxis_title="Date",
            yaxis_title="Stock Price (Close)",
            legend_title="Legend",
            template="plotly_dark"
        )

        # Display in Streamlit
        st.plotly_chart(ma_fig)

        # Display metrics in Streamlit
        st.write(f'**Mean Absolute Error (MAE):** {three_mae:.4f}')
        st.write(f'**Mean Squared Error (MSE):** {three_mse:.4f}')
        st.write(f'**Root Mean Squared Error (RMSE):** {three_rmse:.4f}')

        # Display Future Predictions Table
        st.write("### Future Predictions Table (Next 10 Days)")
        future_df["Date"] = pd.to_datetime(future_df["Date"])
        future_df["Date"] = future_df["Date"].dt.date
        st.table(future_df)

    elif model == "7-Day Moving Average":
        st.write("7-DAY MOVING AVERAGE")

        # Convert Date to datetime and sort
        madf["Date"] = pd.to_datetime(madf["Date"])
        madf = madf.sort_values(by="Date")

        # Moving average features
        madf["Close_3MA"] = madf["Close"].rolling(window=3).mean()
        madf["Close_7MA"] = madf["Close"].rolling(window=7).mean()
        madf["Close_14MA"] = madf["Close"].rolling(window=14).mean()
        madf.dropna(inplace=True)

        # Features and target
        ma_X = madf[["Close_3MA", "Close_7MA", "Close_14MA"]]
        ma_Y = madf["Close"]
        dates = madf["Date"]
        split_date = "2024-01-01"

        # Train-test split
        ma_X_train = ma_X[madf["Date"] < split_date]
        ma_Y_train = ma_Y[madf["Date"] < split_date]
        ma_X_test = ma_X[madf["Date"] >= split_date]
        ma_Y_test = ma_Y[madf["Date"] >= split_date]
        dates_test = dates[madf["Date"] >= split_date]

        # Use the last known Close_7MA as predictions
        ma_Y_pred = ma_X_test["Close_7MA"]

        # Future Predictions (Next 10 Days) using 7-day moving average
        future_dates = pd.date_range(start=dates_test.max(), periods=11, freq="D")[1:]

        # Initialize with last 7 known values
        future_values = list(ma_Y.iloc[-7:].values)

        for _ in range(10):
            next_pred = np.mean(future_values[-7:])  # Compute 7-day moving average
            future_values.append(next_pred)

        # Get only the next 10 predictions
        future_preds_7MA = future_values[-10:]

        # Create DataFrame for Future Predictions
        future_df_7MA = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close (7MA)": future_preds_7MA
        })

        # Evaluate Model
        seven_mae = metrics.mean_absolute_error(ma_Y_test, ma_Y_pred)
        seven_mse = metrics.mean_squared_error(ma_Y_test, ma_Y_pred)
        seven_rmse = np.sqrt(seven_mse)

        # Interactive Plotly Chart
        ma_fig = go.Figure()

        # Train Data
        ma_fig.add_trace(go.Scatter(
            x=dates[madf["Date"] < split_date], 
            y=ma_Y_train, 
            mode='lines', 
            name='Train Data',
            line=dict(color='blue')
        ))

        # Test Data (Actual)
        ma_fig.add_trace(go.Scatter(
            x=dates_test, 
            y=ma_Y_test, 
            mode='lines', 
            name='Test Data (Actual)',
            line=dict(color='green')
        ))

        # Test Data Predictions (7MA)
        ma_fig.add_trace(go.Scatter(
            x=dates_test, 
            y=ma_Y_pred, 
            mode='lines', 
            name='Predictions (7MA)',
            line=dict(color='red', dash='dot')
        ))

        # Future Predictions (7MA)
        ma_fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_preds_7MA, 
            mode='lines', 
            name='Future Predictions (Next 10 Days, 7MA)',
            line=dict(color='orange', dash='dash')
        ))

        # Customize layout
        ma_fig.update_layout(
            title="Seven-Day Moving Average Model Predictions",
            xaxis_title="Date",
            yaxis_title="Stock Price (Close)",
            legend_title="Legend",
            template="plotly_dark"
        )

        # Display in Streamlit
        st.plotly_chart(ma_fig)

        # Display metrics in Streamlit
        st.write(f'**Mean Absolute Error (MAE):** {seven_mae:.4f}')
        st.write(f'**Mean Squared Error (MSE):** {seven_mse:.4f}')
        st.write(f'**Root Mean Squared Error (RMSE):** {seven_rmse:.4f}')

        # Display Future Predictions Table
        st.write("### Future Predictions Table (Next 10 Days, 7MA)")
        future_df_7MA["Date"] = pd.to_datetime(future_df_7MA["Date"])
        future_df_7MA["Date"] = future_df_7MA["Date"].dt.date
        st.table(future_df_7MA)

    elif model == "14-Day Moving Average":
        st.write("14 Day MA")
        madf["Date"] = pd.to_datetime(madf["Date"])
        madf = madf.sort_values(by="Date")

        # Moving average features
        madf["Close_3MA"] = madf["Close"].rolling(window=3).mean()
        madf["Close_7MA"] = madf["Close"].rolling(window=7).mean()
        madf["Close_14MA"] = madf["Close"].rolling(window=14).mean()
        madf.dropna(inplace=True)

        # Features and target
        ma_X = madf[["Close_3MA", "Close_7MA", "Close_14MA"]]
        ma_Y = madf["Close"]
        dates = madf["Date"]
        split_date = "2024-01-01"

        # Train-test split
        ma_X_train = ma_X[madf["Date"] < split_date]
        ma_Y_train = ma_Y[madf["Date"] < split_date]
        ma_X_test = ma_X[madf["Date"] >= split_date]
        ma_Y_test = ma_Y[madf["Date"] >= split_date]
        dates_test = dates[madf["Date"] >= split_date]

        # Use the last known Close_14MA as predictions
        ma_Y_pred = ma_X_test["Close_14MA"]

        # Future Predictions (Next 10 Days) using 14-day moving average
        future_dates = pd.date_range(start=dates_test.max(), periods=11, freq="D")[1:]

        # Initialize with last 14 known values
        future_values = list(ma_Y.iloc[-14:].values)

        for _ in range(10):
            next_pred = np.mean(future_values[-14:])  # Compute 14-day moving average
            future_values.append(next_pred)

        # Get only the next 10 predictions
        future_preds_14MA = future_values[-10:]

        # Create DataFrame for Future Predictions
        future_df_14MA = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close (14MA)": future_preds_14MA
        })

        # Evaluate Model
        fourteen_mae = metrics.mean_absolute_error(ma_Y_test, ma_Y_pred)
        fourteen_mse = metrics.mean_squared_error(ma_Y_test, ma_Y_pred)
        fourteen_rmse = np.sqrt(fourteen_mse)

        # Interactive Plotly Chart
        ma_fig = go.Figure()

        # Train Data
        ma_fig.add_trace(go.Scatter(
            x=dates[madf["Date"] < split_date], 
            y=ma_Y_train, 
            mode='lines', 
            name='Train Data',
            line=dict(color='blue')
        ))

        # Test Data (Actual)
        ma_fig.add_trace(go.Scatter(
            x=dates_test, 
            y=ma_Y_test, 
            mode='lines', 
            name='Test Data (Actual)',
            line=dict(color='green')
        ))

        # Test Data Predictions (14MA)
        ma_fig.add_trace(go.Scatter(
            x=dates_test, 
            y=ma_Y_pred, 
            mode='lines', 
            name='Predictions (14MA)',
            line=dict(color='red', dash='dot')
        ))

        # Future Predictions (14MA)
        ma_fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_preds_14MA, 
            mode='lines', 
            name='Future Predictions (Next 10 Days, 14MA)',
            line=dict(color='orange', dash='dash')
        ))

        # Customize layout
        ma_fig.update_layout(
            title="Fourteen-Day Moving Average Model Predictions",
            xaxis_title="Date",
            yaxis_title="Stock Price (Close)",
            legend_title="Legend",
            template="plotly_dark"
        )

        # Display in Streamlit
        st.plotly_chart(ma_fig)

        # Display metrics in Streamlit
        st.write(f'**Mean Absolute Error (MAE):** {fourteen_mae:.4f}')
        st.write(f'**Mean Squared Error (MSE):** {fourteen_mse:.4f}')
        st.write(f'**Root Mean Squared Error (RMSE):** {fourteen_rmse:.4f}')

        # Display Future Predictions Table
        st.write("### Future Predictions Table (Next 10 Days, 14MA)")
        future_df_14MA["Date"] = pd.to_datetime(future_df_14MA["Date"])
        future_df_14MA["Date"] = future_df_14MA["Date"].dt.date
        st.table(future_df_14MA)  

    elif model == "ARIMA":
        ardf["Date"] = pd.to_datetime(ardf["Date"])
        ardf = ardf.sort_values(by="Date")

        #Features and target
        ar_Y = ardf["Close"]
        dates = ardf["Date"]
        split_date = "2024-01-01"

        # Train-test split
        artrain = ar_Y[ardf["Date"] < split_date]
        artest = ar_Y[madf["Date"] >= split_date]
        dates_train = dates[ardf["Date"] < split_date]
        dates_test = dates[ardf["Date"] >= split_date]

        # Check stationarity
        adf_test = adfuller(artrain)
        #st.write(f"**ADF Test p-value:** {adf_test[1]:.4f} (Stationary if < 0.05)")

        # Optimize ARIMA order
        best_order = auto_arima(artrain, seasonal=False, stepwise=True, trace=True).order
        #st.write(f"**Optimal ARIMA Order:** {best_order}")

        # Train ARIMA model with optimized parameters
        armodel = ARIMA(artrain, order=best_order)
        model_fit = armodel.fit()

        # Predictions on test data (using dynamic forecasting)
        history = artrain.tolist()
        preds = []

        for actual in artest:
            model_dyn = ARIMA(history, order=best_order).fit()
            pred = model_dyn.forecast(steps=1)[0]
            preds.append(pred)
            history.append(actual)  # Update history with actual value

        # Future Predictions (Next 10 Days)
        future_dates = pd.date_range(start=dates_test.max(), periods=11, freq="D")[1:]
        future_preds = model_fit.forecast(steps=10)

        # Create DataFrame for Future Predictions
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": future_preds
        })

        # Evaluate Model
        armae = metrics.mean_absolute_error(artest, preds)
        armse = metrics.mean_squared_error(artest, preds)
        arrmse = np.sqrt(armse)

        #Interactive Plotly Chart
        fig = go.Figure()

        #Train Data
        fig.add_trace(go.Scatter(
            x=dates_train, 
            y=artrain, 
            mode='lines', 
            name='Train Data',
            line=dict(color='blue')
        ))

        #Test Data (Actual)
        fig.add_trace(go.Scatter(
            x=dates_test, 
            y=artest, 
            mode='lines', 
            name='Test Data (Actual)',
            line=dict(color='green')
        ))

        #Predictions on Test Data
        fig.add_trace(go.Scatter(
            x=dates_test, 
            y=preds, 
            mode='lines', 
            name='Predictions',
            line=dict(color='red', dash='dot')
        ))

        #Future Predictions (Next 10 Days)
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_preds, 
            mode='lines', 
            name='Future Predictions (Next 10 Days)',
            line=dict(color='orange', dash='dash')
        ))

        #layout
        fig.update_layout(
            title="ARIMA Model Predictions (Optimized)",
            xaxis_title="Date",
            yaxis_title="Stock Price (Close)",
            legend_title="Legend",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

        # Display metrics in Streamlit
        st.write(f'**Mean Absolute Error (MAE):** {armae:.4f}')
        st.write(f'**Mean Squared Error (MSE):** {armse:.4f}')
        st.write(f'**Root Mean Squared Error (RMSE):** {arrmse:.4f}')

        #Future Predictions Table
        st.write("### Future Predictions Table (Next 10 Days)")

        future_df["Date"] = pd.to_datetime(future_df["Date"])
        future_df["Date"] = future_df["Date"].dt.date
        st.table(future_df)

    else:
        st.write("You have not selected a model")

with tab5:
    st.write("MODEL COMPARISON METRICS")
    st.dataframe(mets)

    st.write("MODEL FORECASTS")
    fpreds["Date"] = pd.to_datetime(fpreds["Date"])
    fpreds["Date"] = fpreds["Date"].dt.date
    st.table(fpreds)


