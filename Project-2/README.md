# Summary of the project:
In this project, a music streaming startup "Sparkify" wants to gain some insight about their user behaviour pattern  through the data they have collected in the following two forms:
1) JSON logs on user activity (resides in S3)
2) JSON metadata of songs in their app (also resides in S3).

The goal here is o build an ETL pipeline that extracts the data from S3, process it using Spark Cluster and loading the processed data back into S3 as set of dimensional tables. The anaytics team can run their queries on processed data to solve their business problems.

# List and Explaination of the files in the project:
`etl.py:` this file fetches the raw data from S3, process it using Spark cluster and uploads back the processed data into S3 again. Running this file will overwrite any existing data in respective output path folders.

`dl.cfg:` this file is currently empty as one needs this file only if someone wants to run the ETL pipeline using his own AWS credentials.

`Readme.md:` It contains project description, process flow and other documentation.

# How to run the Python scripts?

Run `python etl.py` to extract the raw data from S3, process the raw data using Spark clusters and store it back into the S3 for analytics purpose.

# Schema Design
* **Fact Table:** songplays_table
* **Dimension Tables:** users_table, songs_table, artists_table and time_table.
