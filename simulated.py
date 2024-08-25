import random
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
import pandas as pd 
import hts
from hts.hierarchy import HierarchyTree
from  lag_llama_models.forecasting_functions import probabilistic_forecasting
from darts.utils.missing_values import fill_missing_values
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import DLinearModel,NHiTSModel,NLinearModel,RNNModel
from darts.utils.likelihood_models import GaussianLikelihood

import scipy.stats as ss


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("--p",        default=0.1, type=float, help="fraction of outliers")
parser.add_argument("--test_len",        default=20, type=int, help="test length")
parser.add_argument("--input_len",        default=10, type=int, help="input chunk length")
parser.add_argument("--output_len",        default=5, type=int, help="output chunk length")
parser.add_argument("--batch_size",        default=20, type=int, help="output chunk length")
parser.add_argument("--epochs",        default=100, type=int, help="number of epochs")
parser.add_argument("--base_model",        default='DLinear', type=str, help="type of the base forecasting model")



ARGS = parser.parse_args()
print(ARGS)

### Data generation using ARIMA Process

'''  We have a hierarchy with a total number of series n = 7, aggregated series
r = 3 and bottom series m = 4. Each series at the bottom level is generated
using the ARIMA processes with linear trend. Parameters p and q are selected randomly from {0,1,2}.
Error terms for the bottom-level series are drawn from multivariate Gaussian. We consider
 the strong correlation between the series with the same parent and the moderate or small correlation
 between the series with different parents.

We have folowing hieararchical structure
           total   - level 0
          /   \
         A     B   - level 1
       / \    / \
      AA AB BA BB  - level 2 or bottom level
      
      '''
def make_arma(num):
    '''Function generates the samples from ARIMA process
    
    parameter
    -----------
    num : number of sample points we want to generate
    
    '''
    p_q_par = [0,1,2]
    np.random.seed(1001)
    # specify the ARIMA parameters
    p = random.sample(p_q_par,1)[0]  # order of the autoregressive component
    q = random.sample(p_q_par,1)[0]  # order of the moving average component
    if p ==  0 and q == 1 :
        arparams = np.array([0])
        maparams = np.array([0.7])
        arparams = np.r_[1, -arparams]
        maparams = np.r_[1, maparams]
        y = arma_generate_sample(arparams, maparams, num)
        return(y)
    elif p ==  1 and q == 0 :
        arparams = np.array([0.4])
        maparams = np.array([0])
        arparams = np.r_[1, -arparams]
        maparams = np.r_[1, maparams]
        y = arma_generate_sample(arparams, maparams, num)
        return(y)
    elif p ==  1 and q == 1 :
        arparams = np.array([0.4])
        maparams = np.array([0.7])
        arparams = np.r_[1, -arparams]
        maparams = np.r_[1, maparams]
        y = arma_generate_sample(arparams, maparams, num)
        return(y)
    elif p ==  2 and q == 1 :
        arparams = np.array([0.4,0.2])
        maparams = np.array([0.7])
        arparams = np.r_[1, -arparams]
        maparams = np.r_[1, maparams]
        y = arma_generate_sample(arparams, maparams, num)
        return(y)
    elif p == 1 and q == 2 :
        arparams = np.array([0.4])
        maparams = np.array([0.7,0.3])
        arparams = np.r_[1, -arparams]
        maparams = np.r_[1, maparams]
        y = arma_generate_sample(arparams, maparams, num)
        return(y)
    else  :
        arparams = np.array([0.4,0.2])
        maparams = np.array([0.7,0.3])
        arparams = np.r_[1, -arparams]
        maparams = np.r_[1, maparams]
        y = arma_generate_sample(arparams, maparams, num)
        return(y)

np.random.seed(1001)

train_len = 300
test_len = ARGS.test_len
freq = 'D'

total_len = train_len+test_len
split_val = train_len

''' We took the 110 samples from Arima process last 10 observation as test points'''
AA = make_arma(total_len) 
AB = make_arma(total_len)
BA = make_arma(total_len)
BB = make_arma(total_len)


'''Defining mean and covariance matrix to draw residuals from mutlivariate gaussian'''

mean = [5,10,7,8]
cov = [[2,0.8,0.5,0.2], [0.8, 3,0.22,0.23],[0.5,0.22,3,0.7],[0.2,0.23,0.7,3]]

#Drawing residual from multivariate guassian with mean and covariance mentioned above
AA0,AB0,BA0,BB0  = np.random.multivariate_normal(mean, cov, total_len).T

# Adding residuals to the samples 
AA = np.add(AA0,AA)
AB = np.add(AB0,AB)
BA = np.add(BA0,BA)
BB = np.add(BB0,BB)

# defining linear trend for each series

x = np.linspace(0,3,total_len)
AA1 = 10.5*x/2
AB1 = 11*x/2
BA1 = 13.5*x/2
BB1 = 13*x/2

# Adding trend to the series 
AA = np.add(AA1,AA)
AB = np.add(AB1,AB)
BA = np.add(BA1,BA)
BB = np.add(BB1,BB)

''' Defining aggregated series 'total, 'A', 'B' where aggregated series 'A' is the 
sum of series 'AA' and 'AB' and series 'B' is the sum of series 'BA' and 'BB' and the 
most aggregated series 'total' is the sum of 'A' and 'B'  '''
A = np.add(AA,AB)
B = np.add(BA,BB)
total = np.add(A,B)

#creating data frame and adding dates to the data for timse series forecast 
data = {'total': total,'A':A,'B':B, 'AA': AA, 'AB': AB, 'BA':BA,'BB':BB}
df = pd.DataFrame(data)
date = pd.date_range(start='2022/1/1', periods=total_len)
df.insert(0,'date',date)
df = df.set_index('date')

### Creating Hierarchical Levels

''' To use the Darts model, we need to transform the data into the form mentoned in their documentation.
 First, we create the levels according to the hierarchical structure and then 
 define the hierarchy.    
 '''
#levels and hierarchy will vary from data to data 

level0 = [0]        # level 0
level1 = np.array(df.columns[1:3])  # level 1
level1_r = list(range(1,3))    #stored the indices of the series 
level2 = np.array(df.columns[3:7])  # bottom level 
level2_r = list(range(3,7))                    
levels = [level0,level1_r,level2_r]
bottom_level = level2_r
levels1 = [level1,level2]

# Defining hierarchy

total = {'total' : list(level1)}
level1_h = {k: [v for v in level2  if v.startswith(k)] for k in level1}
hierarchy = {**total,**level1_h}
aac = list(hierarchy.keys())   # aggregated series 
aab= list(hierarchy.values())  # child nodes of each aggregated series 

'''Defining aggregation structure that will be needed while geeting the aggregated
forecast using child node forecasts'''
agg_structure1 = []
for i in range(len(aac)):
    for j in range(len(levels1)):
        if aab[i][0] in levels1[j]:
            abcd = []
            for l in aab[i]:
                abcd.append(list(levels1[j]).index(l))
            agg_structure1.append(abcd)


'''Adding bottom level to the aggregation structure'''
for i in range(len(bottom_level)):
    agg_structure1.append([i])


#defining the tree for hierarchy 
ht = HierarchyTree.from_nodes(nodes=hierarchy, df=df)
sum_mat, sum_mat_labels = hts.functions.to_sum_mat(ht) #sum_mat_lables are the all the nodes in the hierarchy
df2 = pd.DataFrame(columns = sum_mat_labels)
for col in sum_mat_labels:
    df2[col] = df[col]
   
#defining the dataframe according to the hierarchy
for col in sum_mat_labels:
    df2[col] = df[col]
    
df2 = df.iloc[: , :len(agg_structure1)]
df2['date'] =  df2.index
df =df2

#Saving test data 
df_test = df2[train_len: len(df2)] 
#Filling the missing values in the data, if any is available 
series_en = fill_missing_values(
    TimeSeries.from_dataframe(
        df, "date"), "auto",)

#Scaling
scaler_en = Scaler()
series_en_transformed = scaler_en.fit_transform(series_en) 
series_en_transformed_copy = series_en_transformed
#Splitting into train and validation 
train_en_transformed, val_en_transformed = series_en_transformed.split_after( 
    pd.Timestamp(df.date[split_val-1]  
) 
)

# add the day as a covariate
day_series = datetime_attribute_timeseries(
    series_en_transformed, attribute="day", one_hot=True
)
scaler_day = Scaler()
day_series = scaler_day.fit_transform(day_series)
train_day, val_day = day_series.split_after(pd.Timestamp(df.date[split_val-1]  
))
# Columns of the data 
columns = train_en_transformed.columns

# origignal data in list format 
df_list = []        
for i in columns:
    df_list.append(df2[i].to_list())
### Fitting the Model

''' Fittting AutoArima model of Darts library 
To know more about Darts: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_arima.html

'''
past_forecast_length = 50 #len(df)- int(len(df)/10) #length of the samples for each period 
backtest_en = []        # predictions for past transformed data 
inv_backtest_en =[]     # predictions for the past data
predictions =[]         # transformed future prediction 
inverse_pred = []       #future prediction 
series_en_transformed = scaler_en.fit_transform(series_en)

if ARGS.base_model=='DLinear':
    model = DLinearModel(
    input_chunk_length=ARGS.input_len,
    output_chunk_length=ARGS.output_len,
    n_epochs=ARGS.epochs,
     batch_size=ARGS.batch_size,
     likelihood=GaussianLikelihood(),
) 
elif ARGS.base_model=='NLinear':
    model = NLinearModel(
    input_chunk_length=ARGS.input_len,
    output_chunk_length=ARGS.output_len,
    n_epochs=ARGS.epochs,
     batch_size=ARGS.batch_size,
     likelihood=GaussianLikelihood(),
)
elif  ARGS.base_model=='NHiTS':
    model = NHiTSModel(
    input_chunk_length=ARGS.input_len,
    output_chunk_length=ARGS.output_len,
    num_blocks=5,
    n_epochs=ARGS.epochs,
    batch_size=150,
    likelihood=GaussianLikelihood()
)
elif  ARGS.base_model=='DeepAR':
    model =RNNModel(
    dropout=0.1,
    batch_size=8,  
    n_epochs=ARGS.epochs,      #100
    n_rnn_layers = 4, #1
    hidden_dim = 20,
    input_chunk_length=ARGS.input_len,
    output_chunk_length = 1,
    likelihood=GaussianLikelihood(),)
    
if ARGS.base_model=='DeepAR':
    model.fit(series=train_en_transformed, verbose=True)
    b1 =model.historical_forecasts(series_en_transformed, 
                                        future_covariates=None,
                                            num_samples=past_forecast_length,
                                            train_length=80, start=0.8, 
                                            forecast_horizon=1, 
                                            stride=1,
                                            retrain=True, 
                                            overlap_end=False,
                                            verbose=True)
else: 
    model.fit(series=train_en_transformed, past_covariates=day_series, verbose=True)
    b1 =model.historical_forecasts(series_en_transformed, 
                                     past_covariates=day_series, 
                                     future_covariates=None,
                                        num_samples=past_forecast_length,
                                        train_length=80, start=0.8, 
                                        forecast_horizon=1, 
                                        stride=1,
                                        retrain=True, 
                                        overlap_end=False,
                                        verbose=True)
    

backtest_en.append(b1)
c1= scaler_en.inverse_transform(b1)
inv_backtest_en.append(c1)
p1 = model.predict(n=test_len,future_covariates=None,num_samples=past_forecast_length)
predictions.append(p1)
inverse_pred.append(scaler_en.inverse_transform(p1))

''' Take samples for the past data and calculate  forecasted mean and standard deviation for past data '''
past=TimeSeries.all_values(c1)
past= np.transpose(past,(1,0,2))
past_m =[]   # 
past_std =[]
for i in range(len(past)):
    aa = np.mean(past[i], axis =1) 
    ab = np.std(past[i],axis=1)
    past_m.append(aa)
    past_std.append(ab)


# plotting the actual and predicted values for the past data 
series_en_transformed = scaler_en.fit_transform(series_en)
series_en_transformed['total'].plot(label="actual")
#backtest_en[list(columns).index('total')].plot(label="predicted", low_quantile=0.01, high_quantile=0.99)
b1['total'].plot(label="predicted", low_quantile=0.01, high_quantile=0.99)

plt.title('Prediction for series: total')
plt.legend()
plt.savefig("Forecast for the series total.png", dpi=300, bbox_inches='tight')

plt.show()

'''Defining the samples for the future period and defining the mean'''

future = TimeSeries.all_values(scaler_en.inverse_transform(p1))
future= np.transpose(future,(1,0,2))
samples = future

'''Defining the standard residuals for the past data., taking mean and sd from past data samples 

     residuals = (actual-mean(predicted))/std(predicted)
     
 We want to store the ranks of these residuals to use in revised bottom-up forecasting. '''
res = []
for i in range(len(past)):
    ab = []
    for j in range(past_forecast_length):
        p = len(df[columns[0]])-past_forecast_length-test_len
        aa = ((df[columns[i]][p+j]-past_m[i][j]))/past_std[i][j]
        ab.append(aa)
    res.append(ab)
    
'''We need sorted samples to re-order accoeding to the ranks of the residuals'''

prob_forecasting = probabilistic_forecasting()

sorted_samples = prob_forecasting.sample_sorting_1(samples) # sample_sorting function is defined in forecasting_funtions.py file 
#defining ranks for the residuals
ranks=[]
for i in range(len(res)):
    ranks.append(ss.rankdata(res[i]))

'''Samples are reorder according to the ranks of the residuals '''
reorderd_samples = prob_forecasting.sample_reordering_3(samples, ranks)


BU_len = sum_mat.shape[1]       #len of the bottom series 
agg_len = len(sum_mat_labels)-BU_len  #length of the aggregated series
agg_samples_index = list(range(0, agg_len))  #indices of the aggregated series 
agg_series_ = columns[0:agg_len]           # names of the aggregated series 

bottom_series = prob_forecasting.getting_bottom_series(samples, sum_mat,sum_mat_labels)   #bottom level without reodering
bottom_series_r = prob_forecasting.getting_bottom_series(reorderd_samples,sum_mat,sum_mat_labels) #bottom series with rank reodering

aggre_series_numbers = agg_structure1     #aggregation structutre 

# bottom series of original observation 
bottom_series_og = prob_forecasting.getting_bottom_series(df_list,sum_mat,sum_mat_labels) #function defined in forecasting_funtions.py file 

### forecat using bottom-up method 

bottom_up_samples_ = prob_forecasting.bottom_up_forecast(columns,bottom_series,bottom_level,aggre_series_numbers,levels) #bottom_up_forecast function defined in forecasting_funtions.py file
bottom_up_samples_ = prob_forecasting.list_reversal(bottom_up_samples_) # list_reversal function defined in forecasting_funtions.py file

# BU forecast with reodering, we reoder the forecasted samples according to the ranks of the past forecast residuals
revised_samples_ = prob_forecasting.bottom_up_revised_forecast(columns,bottom_series_r,bottom_level,aggre_series_numbers,levels,ranks)
revised_samples_ = prob_forecasting.list_reversal(revised_samples_)

### HRA: Bottom up forecast using outlier reordering technique

p = ARGS.p # % of outliers we want to reorder
bottom_up_forecast_h = prob_forecasting.bottom_up_forecast_hueristic(bottom_series, 
                                                                     bottom_series_og,columns,bottom_level,aggre_series_numbers,levels,p)
bottom_up_samples_h = prob_forecasting.list_reversal(bottom_up_forecast_h)

### We use GluonTS library to calculate CRPS and Weighted CRPS 
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx import SimpleFeedForwardEstimator, Trainer
from gluonts.model.forecast import SampleForecast
from gluonts.evaluation import Evaluator
from pandas import Period

custom_dataset = np.array(df_list)
custom_dataset.shape
prediction_length = test_len
freq = freq
start = pd.Period(df.index[0], freq=freq)  # can be different for each time series
# train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
train_ds = ListDataset(
    [{"target": x, "start": start} for x in custom_dataset[:, :-prediction_length]],
    freq=freq,
)
# test dataset: use the whole dataset, add "target" and "start" fields
test_ds = ListDataset(
    [{"target": x, "start": start} for x in custom_dataset], freq=freq
)


quantile = [0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9]

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[4],
    prediction_length=20,
    context_length=20,
    trainer=Trainer(ctx="cpu", epochs=1, learning_rate=1e-5, num_batches_per_epoch=100),
)

predictor = estimator.train(train_ds )
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds ,  # test dataset
    predictor=predictor,  # predictor
    num_samples=60,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)


## Converting our forecast in GluonTS format to calculate errors

start_period = Period(df_test.index[0], 'D')

base_forecast_ = np.transpose(samples,(0,2,1))

bottom_up_samples_ = np.transpose(bottom_up_samples_,(0,2,1))

revised_samples_ = np.transpose(revised_samples_,(0,2,1))

bottom_up_samples_h = np.transpose(bottom_up_samples_h,(0,2,1))

# Base Forecast 
base_forecast = [SampleForecast(samples=samples, start_date=start_period) for samples in base_forecast_]


# BU samples 
BU_forecast = [SampleForecast(samples=samples, start_date=start_period) for samples in bottom_up_samples_]


# revised samples
r_forecast = [SampleForecast(samples=samples, start_date=start_period) for samples in revised_samples_]


# HRA samples
h_forecast = [SampleForecast(samples=samples, start_date=start_period) for samples in bottom_up_samples_h]


# Create an empty DataFrame
loss_df = pd.DataFrame(columns=['Method', 'CRPS', 'Weighted CRPS'])
evaluator = Evaluator(quantile)
agg_metrics, item_metrics = evaluator(tss, base_forecast)
new_row = pd.Series({'Method': 'Base_forecast', 'CRPS': agg_metrics['mean_absolute_QuantileLoss'], 'Weighted CRPS': agg_metrics['mean_wQuantileLoss']})
loss_df.loc[len(loss_df)] = new_row

evaluator = Evaluator(quantile)
agg_metrics, item_metrics = evaluator(tss, BU_forecast)
new_row = pd.Series({'Method': 'BU_forecast', 'CRPS': agg_metrics['mean_absolute_QuantileLoss'], 'Weighted CRPS': agg_metrics['mean_wQuantileLoss']})
loss_df.loc[len(loss_df)] = new_row

evaluator = Evaluator(quantile)
agg_metrics, item_metrics = evaluator(tss, r_forecast)
new_row = pd.Series({'Method': 'Revised_forecast', 'CRPS': agg_metrics['mean_absolute_QuantileLoss'], 'Weighted CRPS': agg_metrics['mean_wQuantileLoss']})
loss_df.loc[len(loss_df)] = new_row

evaluator =Evaluator(quantile)
agg_metrics, item_metrics = evaluator(tss, h_forecast)
new_row = pd.Series({'Method': 'HRA_forecast', 'CRPS': agg_metrics['mean_absolute_QuantileLoss'], 'Weighted CRPS': agg_metrics['mean_wQuantileLoss']})
loss_df.loc[len(loss_df)] = new_row

loss_df.set_index('Method', inplace=True)

print(loss_df)

