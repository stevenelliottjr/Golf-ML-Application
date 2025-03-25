# Installation and Setup Guide

This guide provides step-by-step instructions to set up and run the PGA Tour Performance Analytics Streamlit application.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Step 1: Set Up Your Environment

First, create a directory for your project and navigate to it:

```bash
mkdir golf_analytics
cd golf_analytics
```

## Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment helps manage dependencies for your project separately from your system Python installation.

### For macOS/Linux:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### For Windows:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

Your command prompt should now show the virtual environment name, indicating it's active.

## Step 3: Install Dependencies

With your virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Step 4: Prepare Your Data File

Ensure your `golf_data.csv` file is in the project directory. If your file has a different name or is in a different location, you'll need to update the path in the application code.

## Step 5: Save the Application Code

Copy the entire application code into a file named `app.py` in your project directory.

## Step 6: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application should automatically open in your default web browser. If it doesn't, you can access it at http://localhost:8501.

## Troubleshooting Common Issues

### ImportError with Library Versions

If you encounter import errors, try installing specific versions of libraries:

```bash
pip install streamlit==1.30.0 pandas==2.1.1 numpy==1.26.0
```

### Data Loading Issues

If the application fails to load your data:

1. Check that the file path is correct
2. Verify your CSV file has the expected format and column names
3. Try running a simple test to confirm your CSV is readable:

```python
import pandas as pd
df = pd.read_csv('golf_data.csv')
print(df.head())
```

### Display/UI Issues

If the application runs but doesn't display correctly:

1. Try a different browser
2. Clear your browser cache
3. Ensure you have the latest version of Streamlit

## Customizing the Application

- To change the layout, modify the `st.set_page_config` parameters
- To use a different dataset, update the file path in `load_data()`
- To add new features, consider adding new pages to the sidebar navigation

## Getting Help

If you encounter issues not covered in this guide, check the Streamlit documentation or create an issue in the project repository.