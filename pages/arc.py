import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from lazypredict.Supervised import LazyClassifier
# from lazypredict.Supervised import LazyRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to load and display the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to display the summary of the dataset
def display_summary(df, df2):
    st.header("Dataset")
    st.dataframe(df)
    summary = df2.describe(include='all')
    with st.expander("Dataset Summary",expanded=False):
        st.subheader("Dataset Summary")
        display_summary2(df)
    st.subheader("Numerical variable Summary")
    st.write(summary)

# Function to display the data analysis page
def display_data_analysis(df, df2, df3, df4):
    st.title("Data Analysis Page")
    st.header("Plots")
    # Plot types
    plot_types = ["Line Plot", "Bar Plot", "Scatter Plot", "Pie Chart", "Count Plot"]
    selected_plot_type = st.selectbox("Select Plot Type", plot_types, key="plot_type")
    # Select X-axis column for plotting
    df.dropna(inplace=True)
    
    # Plot the selected columns
    if selected_plot_type == "Line Plot":
        left, middle, right = st.columns(3)
        with left:
            x_column = st.selectbox("Select X-axis column", df.columns, key="x_column")
        with middle:
            y_column = st.selectbox("Select Y-axis column", df2.columns[::-1], key="y_column")
        st.subheader(f"Line Plot for {x_column}")
        fig, ax = plt.subplots()
        with right:
            color1 = st.color_picker('Pick A Color', '#00f900')
        df.plot(x=x_column, y=y_column, kind="line", ax=ax, color=color1)
        st.pyplot(fig)

    elif selected_plot_type == "Bar Plot":
        left, middle, right = st.columns(3)
        with left:
            x_column = st.selectbox("Select X-axis column", df3.columns, key="x_column")
        with middle:
            y_column = st.selectbox("Select Y-axis column", df2.columns[::-1], key="y_column")
        with right:
            color1 = st.color_picker('Pick A Color', '#00f900')
        left2, right2 = st.columns([5, 1])
        with left2:
            aggregation = st.selectbox("Select aggregation", ["Sum", "Average", "Minimum", "Maximum"])
        # Compute the aggregated values
        if aggregation == "Sum":
            aggregated_values = df3.groupby(x_column)[y_column].sum()
        elif aggregation == "Average":
            aggregated_values = df3.groupby(x_column)[y_column].mean()
        elif aggregation == "Minimum":
            aggregated_values = df3.groupby(x_column)[y_column].min()
        elif aggregation == "Maximum":
            aggregated_values = df3.groupby(x_column)[y_column].max()
        sorted_df = df3.sort_values(by=y_column)
        max_values = len(sorted_df)
        with right2:
            num_values = st.number_input(" ", value=10, min_value=1, max_value=max_values)
        top_values = aggregated_values.nlargest(num_values)
        st.subheader(f"Bar Plot for {x_column}")
        st.subheader("TOP " + str(num_values))
        fig, ax = plt.subplots()
        top_values.plot(x=x_column, y=y_column, kind="bar", ax=ax, color=color1)
        st.pyplot(fig)
        st.subheader("TOP " + str(num_values) + " " + x_column + " with " + aggregation + " " + y_column)
        st.write(top_values)

    elif selected_plot_type == "Scatter Plot":
        left, middle, right = st.columns(3)
        with left:
            x_column = st.selectbox("Select X-axis column", df.columns, key="x_column")
        with middle:
            y_column = st.selectbox("Select Y-axis column", df2.columns[::-1], key="y_column")
        fig, ax = plt.subplots()
        hu = st.selectbox("Select column to group", df4.columns, key="hu_column")
        show_regression_line = st.checkbox("Display Regression Line")
        if show_regression_line:
            with right:
                left2, right2 = st.columns(2)
                with left2:
                    colormap1 = st.selectbox("Select colormap", plt.colormaps())
                with right2:
                    line_color = st.color_picker("Select regression line color")
                
            # Calculate regression line
            import numpy as np
            coefficients = np.polyfit(df[x_column], df[y_column], 1)
            m = coefficients[0]  # Slope
            b = coefficients[1]  # Intercept
            regression_line = f"y = {m:.2f}x + {b:.2f}"

            # Display the equation of the regression line
            st.subheader("Regression Line Equation:")
            st.write(regression_line)
            df4.plot(x=x_column, y=y_column, kind="scatter", ax=ax, c=hu, cmap=colormap1)
            # Plot the regression line
            plt.plot(df4[x_column], m * df4[x_column] + b, color=line_color)
        else:
            with right:
                color1 = st.selectbox("Select colormap", plt.colormaps())
            df4.plot(x=x_column, y=y_column, kind="scatter", ax=ax, c=hu, cmap=color1, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = df3[hu].unique()  # Custom hue labels
        ax.legend(handles, new_labels, title='Hue Legend')
        st.pyplot(fig)

    elif selected_plot_type == "Pie Chart":
        x_column = st.selectbox("Select X-axis column", df3.columns, key="x_column")
        if x_column:
            st.subheader(f"Pie Chart for {x_column}")
            max_values = len(df3[x_column].unique())
            num_values = st.slider("", min_value=1, max_value=max_values, value=10, format="%.0f")
            st.subheader("TOP " + str(num_values))
            top_values = df3[x_column].value_counts().head(num_values)
            # Plot the pie chart
            fig, ax = plt.subplots()
            colormap1 = st.selectbox("Select colormap", plt.colormaps())
            top_values.plot(kind="pie", autopct="%1.1f%%", ax=ax, colormap=colormap1)
            ax.set_ylabel("")
            st.pyplot(fig)
    elif selected_plot_type == "Count Plot":
        left, right = st.columns([3, 1])
        with left:
            x_column = st.selectbox("Select X-axis column", df3.columns, key="x_column")
        with right:
            color1 = st.color_picker('Pick A Color', '#00f900')
        if x_column:
            st.subheader(f"Pie Chart for {x_column}")
            max_values = len(df3[x_column].unique())
            num_values = st.slider("", min_value=1, max_value=max_values, value=10, format="%.0f")
            st.subheader("TOP " + str(num_values))
            top_values = df3[x_column].value_counts().head(num_values)
            # Plot the count plot
            fig, ax = plt.subplots()
            top_values.plot(kind="bar", ax=ax, color=color1)
            ax.set_ylabel("")
            st.pyplot(fig)

# Function to perform regression analysis
def perform_regression_analysis(df, dependent_variable, independent_variables):
    # Handle missing values
    df = df.dropna()

    # Handle infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())

    X = df[independent_variables]
    y = df[dependent_variable]

    # Add a constant term to the independent variables
    X = sm.add_constant(X)

    # Fit the OLS regression model
    model = sm.OLS(y, X).fit()

    # Regression statistics
    regression_stats = model.summary()

    # Retrieve the residuals
    residuals = model.resid

    return model, regression_stats, residuals


# Function to display the regression analysis tab
def display_regression_analysis(df):
    st.title("Regression Analysis")
    st.header("Select Variables")
    
    # Select dependent variable
    dependent_variable = st.selectbox("Select Dependent Variable", df.columns)

    # Select multiple independent variables for regression statistics
    independent_variables_regression = st.multiselect("Select Independent Variables for Regression Statistics", df.columns)

    # Perform regression analysis
    if st.button("Perform Regression Analysis"):
        model, regression_stats, residuals = perform_regression_analysis(df, dependent_variable, independent_variables_regression)

        # Store variables in session state
        st.session_state.independent_variables_regression = independent_variables_regression
        st.session_state.residuals = residuals
        
        st.subheader("Regression Statistics")
        st.text(regression_stats)

        # Normality Probability Plot
        if "independent_variables_regression" in st.session_state:
            with st.expander("Normal Probability Plot"):
                fig, ax = plt.subplots()
                stats.probplot(st.session_state.residuals, dist="norm", plot=ax)
                ax.set_xlabel("Theoretical Quantiles")
                ax.set_ylabel("Ordered Values")
                ax.set_title("Normal Probability Plot")
                st.pyplot(fig)

                # Normality conclusion
                alpha = 0.05  # Significance level
                shapiro_test = stats.shapiro(st.session_state.residuals)
                if shapiro_test.pvalue > alpha:
                    st.subheader("Normality Conclusion")
                    st.text("P-value :"+str(shapiro_test.pvalue))
                    st.write(shapiro_test.pvalue)
                    st.write("The residuals appear to follow a normal distribution based on the Normal Probability Plot and Shapiro-Wilk test (p-value > 0.05).")
                else:
                    st.subheader("Normality Conclusion")
                    st.text("P-value :")
                    st.write(shapiro_test.pvalue)
                    st.write("The residuals do not follow a normal distribution based on the Normal Probability Plot and Shapiro-Wilk test (p-value < 0.05).")
        else:
            st.warning("No independent variables selected for regression statistics.")

    
# Function to display the heatmap
def display_heatmap(df):
    st.title("Multivariate Analysis")
    pair = st.checkbox('Plot Pairplot')
    if pair:
        with st.expander("Pair plot", expanded=True):
            st.header("Pair plot")
            plt.figure(figsize=(12, 10))
            sns.pairplot(df)
            st.pyplot()

    st.header("Correlation Heatmap")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    colormaph = st.selectbox("Select colormap for Heat map", plt.colormaps(), key="colormap_heatmap")
    sns.heatmap(correlation_matrix, annot=True, cmap=colormaph)
    st.pyplot()


# Function to display the summary of the dataset
def display_summary2(df):
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    col1.metric("Rows", str(num_rows))
    col2.metric("Columns", str(num_cols))
    
    # Missing values
    st.write("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    col3.metric("Duplicate Rows", str(duplicate_rows))
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    memory_usage_formatted = sizeof_fmt(memory_usage)
    col4.metric("Memory Usage", str(memory_usage_formatted))

    # Number of numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    col5.metric("Numerical Columns", str(len(numerical_cols)))
    col6.metric("Categorical Columns", str(len(categorical_cols)))


# Helper function to format memory size
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f} Y{suffix}"

def extract_data(df):
    # Get min and max rows based on the selected column
    left,right=st.columns(2)
    with left:
        selected_column1 = st.multiselect("Select a column", df[selected_columns].columns)
    with right:
        st.text("Select any or both")
        show_min = st.checkbox("Show Minimum")
        show_max = st.checkbox("Show Maximum")
    for selected_column in selected_column1:
        if show_min:
            min_row = df[df[selected_column] == df[selected_column].min()]
            st.subheader("Minimum Row of "+selected_column)
            st.write(min_row)
        if show_max:
            max_row = df[df[selected_column] == df[selected_column].max()]
            st.subheader("Maximum Row of "+selected_column)
            st.write(max_row)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    tab1, tab2, tab3, tab4= st.tabs(["Home", "Data Analysis", "Multivariate Analysis", "Regression Analysis"])
    # Initialize the dataframe
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df = load_dataset(uploaded_file)
    non_numeric_columns = df.select_dtypes(exclude=['float', 'int']).columns
    df2 = df.drop(non_numeric_columns, axis=1)
    categorical_columns = []
    df3 = pd.DataFrame(df)
    for column in df3.columns:
        if df3[column].dtype == 'object':
            categorical_columns.append(column)
    # Perform label encoding for categorical columns
    for column in categorical_columns:
        df3[column] = pd.factorize(df3[column])[0]
    # Select columns for analysis
    default_columns = list(df2.columns)
    st.sidebar.title("ARC")
    st.sidebar.write("uncover patterns and relationships in your data\n\n\n\n")
    selected_columns = st.sidebar.multiselect("Select columns for analysis", df2.columns, default=default_columns)
    # Display the selected page based on the menu selection
    with tab1:
        st.title("Home")
        if not df.empty:
            display_summary(df, df2[selected_columns])
            extract_data(df)

    with tab2:
        if not df.empty:
            display_data_analysis(df2, df2[selected_columns], df, df3)

    with tab3:
        if not df.empty:
            display_heatmap(df[selected_columns])

    with tab4:
        if not df.empty:
            display_regression_analysis(df[selected_columns])
            
else:
    st.title("Data Analyzer")
    st.subheader("Uncover patterns and relationships in your data")
    st.write("Please upload the file for Analysis")