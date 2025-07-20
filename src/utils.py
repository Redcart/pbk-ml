import os
import pandas as pd
from google.cloud import bigquery
from prophet import Prophet

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INPUT_DATASET = os.getenv("INPUT_DATASET")
BUCKET_NAME = os.getenv("BUCKET_NAME")


def load_table_to_df(sql_query: str) -> pd.DataFrame:
    """
    Load a table from a BigQuery table.

    Args:
        table_path (str): The path to the table file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """

    client = bigquery.Client(project=GCP_PROJECT_ID)
    df = client.query(query=sql_query).result().to_dataframe()
    return df


def save_df_to_gcs(df: pd.DataFrame, gcs_path: str) -> None:
    """
    Save a DataFrame to Google Cloud Storage.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        gcs_path (str): The GCS path where the DataFrame will be saved.
    """

    df.to_csv(gcs_path, index=False)
    print(f"DataFrame saved to {gcs_path}")
    return None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    # Example preprocessing step: drop rows with any NaN values
    df = df.dropna()
    # Rename columns to fit Prophet's requirements
    print("DataFrame preprocessed.")
    return df


def train_prophet_model(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for training.

    Returns:
        Prophet: The trained Prophet model.
    """

    # Rename columns to fit Prophet's requirements
    df = df.rename(
        columns={"ingestion_time": "ds", "sum_nb_bikes_available_mechanic": "y"}
    )
    # Ensure the 'ds' column is in datetime format
    print(df["ds"].dtype)
    print(df.head)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    model = Prophet()
    model.fit(df)
    print("Prophet model trained.")
    return model


def predict_future(model: Prophet, periods: int) -> pd.DataFrame:
    """
    Predict future values using the trained Prophet model.

    Args:
        model (Prophet): The trained Prophet model.
        periods (int): The number of periods to predict.

    Returns:
        pd.DataFrame: The DataFrame containing the predictions.
    """

    future = model.make_future_dataframe(
        periods=periods, freq="5min", include_history=False
    )
    forecast = model.predict(df=future)
    print("Future predictions made.")
    return forecast


def load_df_to_table(df: pd.DataFrame, table_path: str) -> None:
    """
    Load a DataFrame to a BigQuery table.

    Args:
        df (pd.DataFrame): The DataFrame to load.
        table_path (str): The path to the BigQuery table.
    """

    client = bigquery.Client(project=GCP_PROJECT_ID)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_path, job_config=job_config)
    job.result()  # Wait for the job to complete
    print(f"DataFrame loaded to {table_path}")


def run():
    # Example SQL query to load data from BigQuery
    sql_query = f"""
        SELECT
            sum_nb_bikes_available_mechanic,
            sum_nb_bikes_available_electric,
            sum_nb_bikes_available,
            ingestion_time
        FROM `{GCP_PROJECT_ID}.{INPUT_DATASET}.nb_bikes_aggregated`
    """

    # Load data into DataFrame
    df = load_table_to_df(sql_query)
    if df.empty:
        print("No data loaded from BigQuery.")
        return

    # Preprocess the DataFrame
    df = preprocess_data(df)

    # Train the Prophet model
    model = train_prophet_model(df)

    # Save the trained model to GCS (example path)
    gcs_path = f"gs://{BUCKET_NAME}/prophet_model.pkl"
    save_df_to_gcs(df, gcs_path)

    # Predict future values
    forecast = predict_future(model, periods=6 * 12)
    # Save the forecast to GCS (example path)
    forecast_gcs_path = f"gs://{BUCKET_NAME}/forecast.csv"
    save_df_to_gcs(forecast, forecast_gcs_path)

    # Load the DataFrame back to BigQuery
    table_path = "pbk_ml_data.aggregated_capacity_forecast"
    load_df_to_table(forecast, table_path)
    print("Data processing and model training completed.")
    return None


if __name__ == "__main__":
    run()
    print("Run completed successfully.")
