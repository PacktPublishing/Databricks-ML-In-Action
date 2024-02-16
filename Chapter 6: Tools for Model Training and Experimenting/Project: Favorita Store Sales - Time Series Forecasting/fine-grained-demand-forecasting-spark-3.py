# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC For this exercise, we will make use of an increasingly popular library for demand forecasting, [FBProphet](https://facebook.github.io/prophet/), which we will load into the notebook session associated with a cluster running Databricks 7.1 or higher:
# MAGIC
# MAGIC **UPDATE** With Databricks 7.1, we can now install [notebook-scoped libraries](https://docs.databricks.com/dev-tools/databricks-utils.html#library-utilities) using the %pip magic command. 

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()
training_features = "training_dataset"
raw_data_table = "train_set"
label_name = "sales"
time_column = "date"

# COMMAND ----------

raw_data = sql(f"SELECT date, store_nbr, sum(sales) as store_sales FROM {raw_data_table} GROUP BY ALL")
raw_data.createOrReplaceTempView("prophet_train")
display(raw_data)

# COMMAND ----------

# DBTITLE 1,View Weekday Trends
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   dayofweek(date) as weekday,
# MAGIC   AVG(store_sales) as store_sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(store_sales) as store_sales
# MAGIC   FROM prophet_train
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# DBTITLE 1,Retrieve Data for a Single Item-Store Combination
# query to aggregate data to date (ds) level
sql_statement = '''
  SELECT
    CAST(date as date) as ds,
    store_sales as y
  FROM prophet_train
  WHERE store_nbr==1
  ORDER BY ds
  '''

# assemble dataset in Pandas dataframe
history_pd = spark.sql(sql_statement).toPandas()

# drop any missing records
history_pd = history_pd.dropna()

# COMMAND ----------

# MAGIC %md Now, we will import the fbprophet library, but because it can be a bit verbose when in use, we will need to fine-tune the logging settings in our environment:

# COMMAND ----------

# DBTITLE 1,Import Prophet Library
from prophet import Prophet
import logging

# disable informational messages from fbprophet
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,Train Prophet Model
# set model parameters
model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )

# fit the model to historical data
model.fit(history_pd)

# COMMAND ----------

# MAGIC %md Now that we have a trained model, let's use it to build a 90-day forecast:

# COMMAND ----------

# DBTITLE 1,Build Forecast
# define a dataset including both historical dates & 90-days beyond the last available date
future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )

# predict over the dataset
forecast_pd = model.predict(future_pd)

display(forecast_pd)

# COMMAND ----------

# DBTITLE 1,View Historicals vs. Predictions
predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='store_sales')

# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)

display(predict_fig)

# COMMAND ----------

# MAGIC %md **NOTE** This visualization is a bit busy. Bartosz Mikulski provides [an excellent breakdown](https://www.mikulskibartosz.name/prophet-plot-explained/) of it that is well worth checking out.  In a nutshell, the black dots represent our actuals with the darker blue line representing our predictions and the lighter blue band representing our (95%) uncertainty interval.

# COMMAND ----------

# MAGIC %md Visual inspection is useful, but a better way to evaulate the forecast is to calculate Mean Absolute Error, Mean Squared Error and Root Mean Squared Error values for the predicted relative to the actual values in our set:
# MAGIC
# MAGIC **UPDATE** A change in pandas functionality requires us to use *pd.to_datetime* to coerce the date string into the right data type.

# COMMAND ----------

# DBTITLE 1,Calculate Evaluation metrics
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

# get historical actuals & predictions for comparison
actuals_pd = history_pd['y']
predicted_pd = model.predict(history_pd)['yhat']

# calculate evaluation metrics
mae = mean_absolute_error(actuals_pd, predicted_pd)
mse = mean_squared_error(actuals_pd, predicted_pd)
rmse = sqrt(mse)

# print metrics to the screen
print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------

# MAGIC %md FBProphet provides [additional means](https://facebook.github.io/prophet/docs/diagnostics.html) for evaluating how your forecasts hold up over time. You're strongly encouraged to consider using these and those additional techniques when building your forecast models but we'll skip this here to focus on the scaling challenge.

# COMMAND ----------

# MAGIC %md ## Step 3: Scale Forecast Generation
# MAGIC
# MAGIC With the mechanics under our belt, let's now tackle our original goal of building numerous, fine-grain models & forecasts for individual store and item combinations.  We will start by assembling sales data at the store-item-date level of granularity:
# MAGIC
# MAGIC **NOTE**: The data in this data set should already be aggregated at this level of granularity but we are explicitly aggregating to ensure we have the expected data structure.

# COMMAND ----------

# DBTITLE 1,Retrieve Data for All Store-Item Combinations
sql_statement = '''
  SELECT
    store_nbr,
    CAST(date as date) as ds,
    SUM(store_sales) as y
  FROM prophet_train
  GROUP BY store_nbr, ds
  ORDER BY store_nbr, ds
  '''

store_item_history = (
  spark
    .sql( sql_statement )
    .repartition(sc.defaultParallelism, ['store_nbr'])
  ).cache()

# COMMAND ----------

# MAGIC %md With our data aggregated at the store-item-date level, we need to consider how we will pass our data to FBProphet. If our goal is to build a model for each store and item combination, we will need to pass in a store-item subset from the dataset we just assembled, train a model on that subset, and receive a store-item forecast back. We'd expect that forecast to be returned as a dataset with a structure like this where we retain the store and item identifiers for which the forecast was assembled and we limit the output to just the relevant subset of fields generated by the Prophet model:

# COMMAND ----------

# DBTITLE 1,Define Function to Train Model & Generate Forecast
def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # TRAIN MODEL AS BEFORE
  # --------------------------------------
  # remove missing values (more likely at day-store-item level)
  history_pd = history_pd.dropna()
  
  # configure the model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # train the model
  model.fit( history_pd )
  # --------------------------------------
  
  # BUILD FORECAST AS BEFORE
  # --------------------------------------
  # make predictions
  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # get relevant fields from history
  h_pd = history_pd[['ds','store','item','y']].set_index('ds')
  
  # join history and forecast
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # get store & item from incoming data set
  results_pd['store'] = history_pd['store'].iloc[0]
  results_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return results_pd[ ['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

# COMMAND ----------

# MAGIC %md There's a lot taking place within our function, but if you compare the first two blocks of code within which the model is being trained and a forecast is being built to the cells in the previous portion of this notebook, you'll see the code is pretty much the same as before. It's only in the assembly of the required result set that truly new code is being introduced and it consists of fairly standard Pandas dataframe manipulations.

# COMMAND ----------

# MAGIC %md Now let's call our pandas function to build our forecasts.  We do this by grouping our historical dataset around store and item.  We then apply our function to each group and tack on today's date as our *training_date* for data management purposes:
# MAGIC
# MAGIC **UPDATE** Per the previous update note, we are now using applyInPandas() to call a pandas function instead of a pandas UDF.

# COMMAND ----------

# DBTITLE 1,Apply Forecast Function to Each Store-Item Combination
from pyspark.sql.functions import current_date

results = (
  store_item_history
    .groupBy('store', 'item')
      .applyInPandas(forecast_store_item, schema=result_schema)
    .withColumn('training_date', current_date() )
    )

results.createOrReplaceTempView('new_forecasts')

display(results)

# COMMAND ----------

# MAGIC %md We we are likely wanting to report on our forecasts, so let's save them to a queriable table structure:

# COMMAND ----------

# DBTITLE 1,Persist Forecast Output
# MAGIC %sql
# MAGIC -- create forecast table
# MAGIC create table if not exists forecasts (
# MAGIC   date date,
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   sales float,
# MAGIC   sales_predicted float,
# MAGIC   sales_predicted_upper float,
# MAGIC   sales_predicted_lower float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC
# MAGIC -- load data to it
# MAGIC insert into forecasts
# MAGIC select 
# MAGIC   ds as date,
# MAGIC   store,
# MAGIC   item,
# MAGIC   y as sales,
# MAGIC   yhat as sales_predicted,
# MAGIC   yhat_upper as sales_predicted_upper,
# MAGIC   yhat_lower as sales_predicted_lower,
# MAGIC   training_date
# MAGIC from new_forecasts;

# COMMAND ----------

# MAGIC %md But how good (or bad) is each forecast?  Using the pandas function technique, we can generate evaluation metrics for each store-item forecast as follows:

# COMMAND ----------

# DBTITLE 1,Apply Same Techniques to Evaluate Each Forecast
# schema of expected result set
eval_schema =StructType([
  StructField('training_date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('mae', FloatType()),
  StructField('mse', FloatType()),
  StructField('rmse', FloatType())
  ])

# define function to calculate metrics
def evaluate_forecast( evaluation_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # get store & item in incoming data set
  training_date = evaluation_pd['training_date'].iloc[0]
  store = evaluation_pd['store'].iloc[0]
  item = evaluation_pd['item'].iloc[0]
  
  # calulate evaluation metrics
  mae = mean_absolute_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  mse = mean_squared_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  rmse = sqrt( mse )
  
  # assemble result set
  results = {'training_date':[training_date], 'store':[store], 'item':[item], 'mae':[mae], 'mse':[mse], 'rmse':[rmse]}
  return pd.DataFrame.from_dict( results )

# calculate metrics
results = (
  spark
    .table('new_forecasts')
    .filter('ds < \'2018-01-01\'') # limit evaluation to periods where we have historical data
    .select('training_date', 'store', 'item', 'y', 'yhat')
    .groupBy('training_date', 'store', 'item')
    .applyInPandas(evaluate_forecast, schema=eval_schema)
    )

results.createOrReplaceTempView('new_forecast_evals')

# COMMAND ----------

# MAGIC %md Once again, we will likely want to report the metrics for each forecast, so we persist these to a queriable table:

# COMMAND ----------

# DBTITLE 1,Persist Evaluation Metrics
# MAGIC %sql
# MAGIC
# MAGIC create table if not exists forecast_evals (
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   mae float,
# MAGIC   mse float,
# MAGIC   rmse float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC
# MAGIC insert into forecast_evals
# MAGIC select
# MAGIC   store,
# MAGIC   item,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse,
# MAGIC   training_date
# MAGIC from new_forecast_evals;

# COMMAND ----------

# MAGIC %md We now have constructed a forecast for each store-item combination and generated basic evaluation metrics for each.  To see this forecast data, we can issue a simple query (limited here to product 1 across stores 1 through 3):

# COMMAND ----------

# DBTITLE 1,Visualize Forecasts
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   store,
# MAGIC   date,
# MAGIC   sales_predicted,
# MAGIC   sales_predicted_upper,
# MAGIC   sales_predicted_lower
# MAGIC FROM forecasts a
# MAGIC WHERE item = 1 AND
# MAGIC       store IN (1, 2, 3) AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md And for each of these, we can retrieve a measure of help us assess the reliability of each forecast:

# COMMAND ----------

# DBTITLE 1,Retrieve Evaluation Metrics
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   store,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse
# MAGIC FROM forecast_evals a
# MAGIC WHERE item = 1 AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store
