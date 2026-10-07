"""ETM+ <-> OLI surface reflectance transformation coefficients.

Roy, D.P. et al. (2016): Characterization of Landsat-7 to Landsat-8 reflective
wavelength and normalized difference vegetation index continuity. Remote Sensing
of Environment 185, 57-70, doi:10.1016/j.rse.2015.12.024, Table 2 (surface
reflectance, OLS and RMA). Reflectance as 0-1, keys are STAC common names.

Caveats:
- the paper corrected both sensors with LEDAPS; Collection 2 uses LaSRC for OLI,
  so a processing offset (mainly blue/green) is not covered by the coefficients
- TM (L4/5) is not in the paper, it is transformed with the ETM+ coefficients
- Landsat 9 (OLI-2) is not in the paper, it is treated as OLI
- fitted on CONUS, summer 2013 and winter 2013/14
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

SOURCE = "Roy et al. 2016, Table 2"


@dataclass(frozen=True)
class Line:
    """y = intercept + slope * x"""

    intercept: float
    slope: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * x

    def inverse(self) -> "Line":
        """x = (y - intercept) / slope, exact only for a symmetric (RMA) fit."""
        return Line(-self.intercept / self.slope, 1 / self.slope)


# OLS, OLI = a + b * ETM+
OLS_ETM_TO_OLI = MappingProxyType(
    {
        "blue": Line(0.0003, 0.8474),
        "green": Line(0.0088, 0.8483),
        "red": Line(0.0061, 0.9047),
        "nir08": Line(0.0412, 0.8462),
        "swir16": Line(0.0254, 0.8937),
        "swir22": Line(0.0172, 0.9071),
    }
)

# OLS, ETM+ = a + b * OLI (separate fit, not the inverse of OLS_ETM_TO_OLI)
OLS_OLI_TO_ETM = MappingProxyType(
    {
        "blue": Line(0.0183, 0.8850),
        "green": Line(0.0123, 0.9317),
        "red": Line(0.0123, 0.9372),
        "nir08": Line(0.0448, 0.8339),
        "swir16": Line(0.0306, 0.8639),
        "swir22": Line(0.0116, 0.9165),
    }
)

# RMA, OLI = a + b * ETM+; symmetric, so OLI -> ETM+ is the exact inverse
RMA_ETM_TO_OLI = MappingProxyType(
    {
        "blue": Line(-0.0095, 0.9785),
        "green": Line(-0.0016, 0.9542),
        "red": Line(-0.0022, 0.9825),
        "nir08": Line(-0.0021, 1.0073),
        "swir16": Line(-0.0030, 1.0171),
        "swir22": Line(0.0029, 0.9949),
    }
)

RMA_OLI_TO_ETM = MappingProxyType(
    {band: line.inverse() for band, line in RMA_ETM_TO_OLI.items()}
)

COEFFICIENTS = MappingProxyType(
    {
        ("ols", "etm_to_oli"): OLS_ETM_TO_OLI,
        ("ols", "oli_to_etm"): OLS_OLI_TO_ETM,
        ("rma", "etm_to_oli"): RMA_ETM_TO_OLI,
        ("rma", "oli_to_etm"): RMA_OLI_TO_ETM,
    }
)
