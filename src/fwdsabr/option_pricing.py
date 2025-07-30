import typing as tp
import numpy as np
import numpy.typing as npt
import scipy.stats as ss
import pandas as pd

call_put_literal = tp.Literal['call', 'put']
def compute_option_prices(
        F: float | npt.NDArray[np.float64] | pd.Series, 
        K: float | npt.NDArray[np.float64] | pd.Series, 
        vol: float | npt.NDArray[np.float64] | pd.Series, 
        T: float | int | npt.NDArray[np.float64] | pd.Series, 
        call_put: call_put_literal | list[call_put_literal]
        ):
    alpha = (K-F) / (vol * np.sqrt(T))
    alpha_pdf = ss.norm.pdf(alpha)
    alpha_cdf = ss.norm.cdf(alpha)
    
    prob_itm = 1 - alpha_cdf
    # prob_itm = np.clip(prob_itm, 1e-9, 1.0 - 1e-9)  # Avoid division by zero
    S_d1 = F + vol * np.sqrt(T) * alpha_pdf / (1 - alpha_cdf)
    S_d1 = np.nan_to_num(S_d1, nan=0.0)  # Handle NaN values gracefully
    price_call = (S_d1 - K) * prob_itm 

    prob_itm = alpha_cdf
    # prob_itm = np.clip(prob_itm, 1e-10, 1.0 - 1e-10)  # Avoid division by zero
    S_d1 = F - vol * np.sqrt(T) * alpha_pdf / alpha_cdf
    S_d1 = np.nan_to_num(S_d1, nan=0.0)  # Handle NaN values gracefully
    price_put = (K - S_d1) * prob_itm

    if isinstance(call_put, str):
        call_put = [call_put]
    
    call_put_multiplier_call = np.array([1 if cp == 'call' else 0 for cp in call_put])
    call_put_multiplier_put = np.array([1 if cp == 'put' else 0 for cp in call_put])
    return price_call * call_put_multiplier_call + price_put * call_put_multiplier_put