import streamlit as st
import pandas as pd

def main():
    st.title("Golf Data Analysis")

    # --- 1. Load the CSV file ---
    # Make sure the file name matches exactly, and that the CSV
    # is in the same directory or you specify a correct path.
    df = pd.read_csv("golf_data.csv")

    # --- 2. Preview the data ---
    st.subheader("Preview of the Dataset")
    st.dataframe(df.head())

    # --- 3. Check for required columns ---
    # Adjust this list if your CSV actually has different names
    required_cols = ["tournament_name", "sg_total"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # If we’re missing columns, stop the app with an error message
        st.error(f"The following required columns are missing in the dataset: {', '.join(missing_cols)}")
        st.stop()

    # Optionally, rename columns to something more convenient
    # (only if you prefer referencing them differently in code)
    df.rename(columns={"tournament_name": "tournament_name"}, inplace=True)

    # --- 4. Simple analysis / grouping example ---
    # (Demonstrates that 'tournament_name' and 'sg_total' now exist)
    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Average sg_total by Tournament")
    avg_sg = df.groupby("tournament_name")["sg_total"].mean().reset_index()
    st.dataframe(avg_sg)

    # --- 5. Any additional logic / visualizations go here ---
    # For example, sorting players, or further calculations, etc.
    # Just make sure you reference the correct column names.

if __name__ == "__main__":
    main()