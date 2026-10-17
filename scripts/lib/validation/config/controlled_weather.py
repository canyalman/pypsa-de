# SPDX-License-Identifier: MIT

from typing import Literal

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from scripts.lib.validation.config._base import ConfigModel


class ControlledWeatherConfig(ConfigModel):
    enable: bool = False
    mode: Literal["capacity_expansion", "dispatch"] = "capacity_expansion"
    dispatch_unserved_energy: bool = True
    dispatch_unserved_energy_cost: PositiveFloat = 100000.0
    nominal_clip_tolerance: NonNegativeFloat = 0.05
    planning_horizon: int = Field(2035, strict=True)
    previous_horizon: int = Field(2030, strict=True)
    previous_network: str = ""
    reference_network: str = ""
    weather_resources_directory: str = ""
    weather_profile_horizons: list[int] = Field(default_factory=lambda: [2025, 2030])
    domestic_country: str = "DE"
    frozen_generator_carriers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self):
        if self.enable:
            if self.planning_horizon != 2035 or self.previous_horizon != 2030:
                raise ValueError(
                    "Controlled weather requires the common 2030 -> 2035 experiment."
                )
            if not self.previous_network.endswith(
                "_2030.nc"
            ) or not self.reference_network.endswith("_2035.nc"):
                raise ValueError(
                    "Explicit 2030 history and 2035 fleet reference networks are required."
                )
            if self.weather_profile_horizons != [2025, 2030]:
                raise ValueError("Fresh 2025/2030-vintage weather inputs are required.")
            if (
                not self.frozen_generator_carriers
                or not self.weather_resources_directory
            ):
                raise ValueError(
                    "Frozen renewable carriers and a weather resources directory are required."
                )
        return self
