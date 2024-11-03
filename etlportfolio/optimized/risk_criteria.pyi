import numpy as np

def compute_portfolio_moments(
    weights: np.ndarray,
    returns: np.ndarray,
    variance: np.ndarray,
) -> tuple[float, float]:
    ...


def compute_lpm_integrand(
    x: float,
    tau: float,
    power: int,
    port_ret: float,
    port_vol: float,
) -> float:
    ...


def compute_cvar_integrand(
    x: float,
    port_ret: float,
    port_vol: float,
) -> float:
    ...
