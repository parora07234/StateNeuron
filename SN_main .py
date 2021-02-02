from typing import List, Optional
import numpy as np
import mxnet as mx
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from itertools import islice
from pathlib import Path
from pandas.tseries.frequencies import to_offset

load_train = pd.read_csv('ISO-NE with temp.csv',parse_dates=[['Date', 'Time']],header=0,index_col=0)
load_train.head()

train=load_train.transpose()
train.head()

train2=train.to_numpy()
type(train2)
train2.shape

feat_static_cat=train2[[0,2,4,6,8,10,12,14],0]
feat_static_cat.shape

feat_dynamic_real=train2[[1,3,5,7,9,11,13,15],:]
feat_dynamic_real.shape

target=train2[[0,2,4,6,8,10,12,14],:]
target.shape

freq='1H'
prediction_length=24
start= [pd.Timestamp("2017-01-01", freq='1H') 
                                for _ in range(8)]


class StateNeuronEstimator:
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        add_trend: bool = False,
        past_length: Optional[int] = None,
        num_periods_to_train: int = 4,
        trainer: Trainer = Trainer(
            epochs=1, num_batches_per_epoch=50, hybridize=False
        ),
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = True,
        embedding_dimension: Optional[List[int]] = None,
        issm: Optional[ISSM] = [CompositeISSM],
        scaling: bool = True,
        time_features: Optional[List[TimeFeature]] = [HourOfDay],
        noise_std_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        prior_cov_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        innovation_bounds: ParameterBounds = ParameterBounds(1e-6, 0.01),
    ) -> None:
        super().__init__(trainer=trainer)
        
        self.freq = freq
        self.past_length = (
            past_length
            if past_length is not None
            else num_periods_to_train * longest_period_from_frequency_str(freq)
        )
        self.prediction_length = prediction_length
        self.add_trend = add_trend
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.scaling = scaling
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.cardinality = (
            cardinality if cardinality and use_feat_static_cat else [1]
        )
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )

        self.issm = (
            issm
            if issm is not None
            else CompositeISSM.get_from_freq(freq, add_trend)
        )

        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.noise_std_bounds = noise_std_bounds
        self.prior_cov_bounds = prior_cov_bounds
        self.innovation_bounds = innovation_bounds
        
estimator=StateNeuronEstimator (freq='H',
        prediction_length=24,
        cardinality=[8],
        add_trend= True,
        num_periods_to_train= 4,
        trainer= StateNeuronTrainer(
            epochs=1, num_batches_per_epoch=128, hybridize=False),
        num_layers= 2,
        num_cells= 40,
        cell_type= "lstm",
        num_parallel_samples = 100,
        dropout_rate= 0.1,
        use_feat_dynamic_real = True,
        use_feat_static_cat = True,
        embedding_dimension = None,
        issm =[CompositeISSM] ,
        scaling = True,
        time_features = [HourOfDay])

predictor = estimator.train(train_ds)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds, 
    predictor=StateNeuronPredictor,  
    num_samples=100, 
)
forecasts = list(forecast_it)
tss = list(ts_it)



