import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek
from pyspark.sql.types import IntegerType, TimestampType as Ts
from pyspark.sql.functions import monotonically_increasing_id

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    This function processes the song data stored on S3
    to create the artists and songs table.
    :spark: a spark session
    :input_data: input file path
    :output_data: output file path
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data/*/*/*/*.json")
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id','title','artist_id','year','duration'])\
                    .dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    path_song = os.path.join(output_data, "songs")
    songs_table.write.parquet(path_song, partitionBy=("year","artist_id"))

    # extract columns to create artists table
    artists_table = df.select(['artist_id','artist_name','artist_location',
                               'artist_latitude', 'artist_longitude'])\
                      .dropDuplicates()
    
    # write artists table to parquet files
    path_artists = os.path.join(output_data, "artists")
    artists_table.write.parquet(path_artists)


def process_log_data(spark, input_data, output_data):
    """
    This function extracts the log data files from the S3 bucket,
    tranforms and writes them into appropriately partitioned parquet 
    tables namely users, time and songplays.
    :spark: a spark session
    :input_data: input file path
    :output_data: output file path
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, "log_data/*/*/*.json")

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.select(['userId','firstName','lastName','gender','level'])\
                    .dropDuplicates()
    
    # write users table to parquet files
    path_users = os.path.join(output_data, "users")
    users_table.write.parquet(path_users)

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x/1000, IntegerType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), Ts())
    df = df.withColumn('datetime', get_datetime(df.timestamp))
    
    # extract columns to create time table
    time_table = df.select(
        col('datetime').alias('start_time'),
        hour('datetime').alias('hour'),
        dayofmonth('datetime').alias('day'),
        weekofyear('datetime').alias('week'),
        month('datetime').alias('month'),
        year('datetime').alias('year'),
        dayofweek('datetime').alias('weekday')
    ).dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    path_tt = os.path.join(output_data, "time")
    time_table.write.parquet(path_tt, partitionBy = ["year","month"])

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data+"songs")

    # extract columns from joined song and log datasets to create songplays table
    df = df.join(song_df, song_df.title == df.song)
    df = df.withColumn('songplay_id', monotonically_increasing_id())
    df = df.withColumn("month", month(df.datetime))
    songplays_table = df['songplay_id', 'year', 'month', 'userId','level', \
                      'song_id', 'artist_id', 'sessionId', 'location', 'userAgent']

    # write songplays table to parquet files partitioned by year and month
    path_songplays = os.path.join(output_data, 'songplays')
    songplays_table.write.parquet(path_songplays, partitionBy=["year","month"])


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3://udacity-scb-proj2/result/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
