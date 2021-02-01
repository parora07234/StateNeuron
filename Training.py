from typing import List, Optional
import mxnet as mx
class StateNeuronNetwork:
    @validated()
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        past_length: int,
        prediction_length: int,
        issm: ISSM,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        scaling: bool = True,
        noise_std_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        prior_cov_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        innovation_bounds: ParameterBounds = ParameterBounds(1e-6, 0.01),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.past_length = past_length
        self.prediction_length = prediction_length
        self.issm = issm
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling

        assert len(cardinality) == len(
            embedding_dimension
        ), "embedding_dimension should be a list with the same size as cardinality"
        self.univariate = self.issm.output_dim() == 1

        self.noise_std_bounds = noise_std_bounds
        self.prior_cov_bounds = prior_cov_bounds
        self.innovation_bounds = innovation_bounds 
    def compute_lds(
        self,
        F,
        feat_static_cat: Tensor,
        seasonal_indicators: Tensor,
        time_feat: Tensor,
        length: int,
        prior_mean: Optional[Tensor] = None,
        prior_cov: Optional[Tensor] = None,
        lstm_begin_state: Optional[List[Tensor]] = None,
    ):       
    lds = LDS(
            emission_coeff=emission_coeff,
            transition_coeff=transition_coeff,
            innovation_coeff=F.broadcast_mul(innovation, innovation_coeff),
            noise_std=noise_std,
            residuals=residuals,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            latent_dim=self.issm.latent_dim(),
            output_dim=self.issm.output_dim(),
            seq_length=length,
        )

        return lds, lstm_final_state
class StateNeuronTrainingNetwork: 
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
        past_seasonal_indicators: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
    ) -> Tensor:
        lds, _ = self.compute_lds(
            F,
            feat_static_cat=feat_static_cat,
            seasonal_indicators=past_seasonal_indicators.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            time_feat=past_time_feat.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            length=self.past_length,
        )

        _, scale = self.scaler(past_target, past_observed_values)

        observed_context = past_observed_values.slice_axis(
            axis=1, begin=-self.past_length, end=None
        )

        ll, _, _ = lds.log_prob(
            x=past_target.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            observed=observed_context.min(axis=-1, keepdims=False),
            scale=scale,
        )

        return weighted_average(
            F=F, x=-ll, axis=1, weights=observed_context.squeeze(axis=-1)
        )