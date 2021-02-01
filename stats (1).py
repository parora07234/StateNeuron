from typing import List, Tuple
from pandas.tseries.frequencies import to_offset

class CompositeISSM(ISSM):
    DEFAULT_ADD_TREND: bool = True

    @validated()
    def __init__(
        self,
        seasonal_issms: List[SeasonalityISSM],
        add_trend: bool = DEFAULT_ADD_TREND,
    ) -> None:
        super(CompositeISSM, self).__init__()
        self.seasonal_issms = seasonal_issms
        self.nonseasonal_issm = (
            LevelISSM() if add_trend is False else LevelTrendISSM()
        )

    def latent_dim(self) -> int:
        return (
            sum([issm.latent_dim() for issm in self.seasonal_issms])
            + self.nonseasonal_issm.latent_dim()
        )

    def output_dim(self) -> int:
        return self.nonseasonal_issm.output_dim()

    @classmethod
    def get_from_freq(cls, freq: str, add_trend: bool = DEFAULT_ADD_TREND):
        offset = to_offset(freq)

        seasonal_issms: List[SeasonalityISSM] = []

        if offset.name == "M":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=12)  # month-of-year seasonality
            ]
        elif offset.name == "W-SUN":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=53)  # week-of-year seasonality
            ]
        elif offset.name == "D":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=7)
            ]  # day-of-week seasonality
        elif offset.name == "B":  # TODO: check this case
            seasonal_issms = [
                SeasonalityISSM(num_seasons=7)
            ]  # day-of-week seasonality
        elif offset.name == "H":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
                SeasonalityISSM(num_seasons=7),  # day-of-week seasonality
            ]
        elif offset.name == "T":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=60),  # minute-of-hour seasonality
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return cls(seasonal_issms=seasonal_issms, add_trend=add_trend)

    @classmethod
    def seasonal_features(cls, freq: str) -> List[TimeFeature]:
        offset = to_offset(freq)
        if offset.name == "M":
            return [MonthOfYear(normalized=False)]
        elif offset.name == "W-SUN":
            return [WeekOfYear(normalized=False)]
        elif offset.name == "D":
            return [DayOfWeek(normalized=False)]
        elif offset.name == "B":  # TODO: check this case
            return [DayOfWeek(normalized=False)]
        elif offset.name == "H":
            return [HourOfDay(normalized=False), DayOfWeek(normalized=False)]
        elif offset.name == "T":
            return [
                MinuteOfHour(normalized=False),
                HourOfDay(normalized=False),
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return []

    def get_issm_coeff(
        self, seasonal_indicators: Tensor  # (batch_size, time_length)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        F = getF(seasonal_indicators)
        emission_coeff_ls, transition_coeff_ls, innovation_coeff_ls = zip(
            self.nonseasonal_issm.get_issm_coeff(seasonal_indicators),
            *[
                issm.get_issm_coeff(
                    seasonal_indicators.slice_axis(
                        axis=-1, begin=ix, end=ix + 1
                    )
                )
                for ix, issm in enumerate(self.seasonal_issms)
            ],
        )
        emission_coeff = F.concat(*emission_coeff_ls, dim=-1)

        innovation_coeff = F.concat(*innovation_coeff_ls, dim=-1)

        transition_coeff = _make_block_diagonal(transition_coeff_ls)

        return emission_coeff, transition_coeff, innovation_coeff
