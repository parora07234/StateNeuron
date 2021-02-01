
from typing import List, Optional

# Standard library imports
import numpy as np


from pandas.tseries.frequencies import to_offset


SEASON_INDICATORS_FIELD = "seasonal_indicators"


# A dictionary mapping granularity to the period length of the longest season
# one can expect given the granularity of the time series.

FREQ_LONGEST_PERIOD_DICT = {
    "M": 12,  # yearly seasonality
    "W-SUN": 52,  # yearly seasonality
    "D": 31,  # monthly seasonality
    "B": 22,  # monthly seasonality
    "H": 168,  # weekly seasonality
    "T": 1440,  # daily seasonality
}


def longest_period_from_frequency_str(freq_str: str) -> int:
    offset = to_offset(freq_str)
    return FREQ_LONGEST_PERIOD_DICT[offset.name] // offset.n


class StateNeuronEstimator:
    """
    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    cardinality
        Number of values of each categorical feature.
        This must be set by default unless ``use_feat_static_cat``
        is set to `False` explicitly (which is NOT recommended).
    add_trend
        Flag to indicate whether to include trend component in the
        state space model
    past_length
        This is the length of the training time series;
        i.e., number of steps to unroll the RNN for before computing 
        predictions.
        Set this to (at most) the length of the shortest time series in the 
        dataset.
        (default: None, in which case the training length is set such that 
        at least
        `num_seasons_to_train` seasons are included in the training.
        See `num_seasons_to_train`)
    num_periods_to_train
        (Used only when `past_length` is not set)
        Number of periods to include in the training time series. (default: 4)
        Here period corresponds to the longest cycle one can expect given 
        the granularity of the time series.
        See: https://stats.stackexchange.com/questions/120806/frequency
        -value-for-seconds-minutes-intervals-data-in-r
    trainer
        Trainer object to be used (default: Trainer())
    num_layers
        Number of RNN layers (default: 2)
    num_cells
        Number of RNN cells for each layer (default: 40)
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism 
        during inference.
        This is a model optimization that does not affect the accuracy (
        default: 100).
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    use_feat_dynamic_real
        Whether to use the ``feat_dynamic_real`` field from the data
        (default: False)
    use_feat_static_cat
        Whether to use the ``feat_static_cat`` field from the data
        (default: True)
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    scaling
        Whether to automatically scale the target values (default: true)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    noise_std_bounds
        Lower and upper bounds for the standard deviation of the observation
        noise
    prior_cov_bounds
        Lower and upper bounds for the diagonal of the prior covariance matrix
    innovation_bounds
        Lower and upper bounds for the standard deviation of the observation 
        noise
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        add_trend: bool = False,
        past_length: Optional[int] = None,
        num_periods_to_train: int = 4,
        trainer: Trainer = Trainer(
            epochs=100, num_batches_per_epoch=50, hybridize=False
        ),
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = True,
        embedding_dimension: Optional[List[int]] = None,
        issm: Optional[ISSM] = None,
        scaling: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
        noise_std_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        prior_cov_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        innovation_bounds: ParameterBounds = ParameterBounds(1e-6, 0.01),
    ) -> None:
        super().__init__(trainer=trainer)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            past_length is None or past_length > 0
        ), "The value of `past_length` should be > 0"
        assert num_layers > 0, "The value of `num_layers` should be > 0"
        assert num_cells > 0, "The value of `num_cells` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"
        assert not use_feat_static_cat or any(c > 1 for c in cardinality), (
            f"Cardinality of at least one static categorical feature must be larger than 1 "
            f"if `use_feat_static_cat=True`. But cardinality provided is: {cardinality}"
        )
        assert embedding_dimension is None or all(
            e > 0 for e in embedding_dimension
        ), "Elements of `embedding_dimension` should be > 0"

        assert all(
            np.isfinite(p.lower) and np.isfinite(p.upper) and p.lower > 0
            for p in [noise_std_bounds, prior_cov_bounds, innovation_bounds]
        ), "All parameter bounds should be finite, and lower bounds should be positive"

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




