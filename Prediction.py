class StateNeuronPredictionNetwork:

    @validated()
    def __init__(self, num_parallel_samples: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
        past_seasonal_indicators: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        future_seasonal_indicators: Tensor,
        future_time_feat: Tensor,
    ) -> Tensor:
        lds, lstm_state = self.compute_lds(
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

        _, final_mean, final_cov = lds.log_prob(
            x=past_target.slice_axis(
                axis=1, begin=-self.past_length, end=None
            ),
            observed=observed_context.min(axis=-1, keepdims=False),
            scale=scale,
        )

        lds_prediction, _ = self.compute_lds(
            F,
            feat_static_cat=feat_static_cat,
            seasonal_indicators=future_seasonal_indicators,
            time_feat=future_time_feat,
            length=self.prediction_length,
            lstm_begin_state=lstm_state,
            prior_mean=final_mean,
            prior_cov=final_cov,
        )

        samples = lds_prediction.sample(
            num_samples=self.num_parallel_samples, scale=scale
        )

        # convert samples from
        # (num_samples, batch_size, prediction_length, target_dim)
        # to
        # (batch_size, num_samples, prediction_length, target_dim)
        # and squeeze last axis in the univariate case
        if self.univariate:
            return samples.transpose(axes=(1, 0, 2, 3)).squeeze(axis=3)
        else:
            return samples.transpose(axes=(1, 0, 2, 3))
