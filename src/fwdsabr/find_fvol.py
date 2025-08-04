import typing
import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import rel_entr
from pysabr import Hagan2002NormalSABR
from fwdsabr.option_pricing import compute_option_prices

def get_interpolated_vols(sabr: Hagan2002NormalSABR):
    space = np.linspace(
        start=-sabr.shift + 0.01 , 
        stop=.2, 
        num=1000
        )
    vols = sabr.normal_vol(space)
    interp_vol = interp1d(space, vols, fill_value='extrapolate') # type: ignore
    return interp_vol

class SabrDist(ss.rv_continuous):
    def __init__(self, f, shift, t, v_atm_n, beta, rho, volvol):
        super().__init__(name='SABR', )
        self.sabr = Hagan2002NormalSABR(
            f=f, shift=shift, 
            t=t, 
            v_atm_n=v_atm_n, 
            beta=beta, 
            rho=rho, 
            volvol=volvol
            )
        self.interp_vol = get_interpolated_vols(self.sabr)
    
    def normal_vol(self, x):
        return self.interp_vol(x)

    def _cdf(self, x): # type: ignore
        vols = self.normal_vol(x)
        return ss.norm.cdf(
            x=x, 
            loc = self.sabr.f, 
            scale = vols * np.sqrt(self.sabr.t)
            )
    
    @classmethod
    def from_sabr(cls, sabr: Hagan2002NormalSABR):
        return cls(
            f=sabr.f, 
            shift=sabr.shift, 
            t=sabr.t, 
            v_atm_n=sabr.v_atm_n, 
            beta=sabr.beta, 
            rho=sabr.rho, 
            volvol=sabr.volvol
            )

def get_total_pdf(
        sabr_p1: Hagan2002NormalSABR, 
        sabr_p1_p2: Hagan2002NormalSABR,
        strikes: np.ndarray
        ):
    sabr_dist_p1 = SabrDist.from_sabr(sabr_p1)
    sabr_dist_p1_p2 = SabrDist.from_sabr(sabr_p1_p2)
    pdf_p1 = sabr_dist_p1.pdf(strikes) / sabr_dist_p1.pdf(strikes).sum()
    
    vols_p1_p2 = sabr_dist_p1_p2.normal_vol(strikes)
    pdf = ss.norm.pdf(
        x=strikes.reshape(-1, 1) + sabr_p1_p2.f, 
        loc = strikes.reshape(1, -1) + sabr_p1_p2.f, 
        scale = vols_p1_p2.reshape(1, -1) * np.sqrt(sabr_p1_p2.t)
    )
    
    pdf = pdf / pdf.sum()
    pdf_total = (pdf * pdf_p1).sum(axis=1)
    return pdf_total

# def find_fvol(sabr_p1, sabr_p2, strikes):
#     def to_optimize(x, strikes, pdf_p2):
#         v_atm_n, shift, beta, rho, volvol = x
#         sabr_p1_p2 = Hagan2002NormalSABR(
#             f=sabr_p2.f, 
#             shift=shift, 
#             t=(sabr_p2.t - sabr_p1.t), 
#             v_atm_n=v_atm_n, 
#             beta=beta, 
#             rho=rho, 
#             volvol=volvol
#             )
#         pdf_total = get_total_pdf(sabr_p1=sabr_p1, sabr_p1_p2=sabr_p1_p2, strikes=strikes)
#         kl = np.sum(rel_entr(pdf_total, pdf_p2)) * 10**6
#         return kl
    
#     def run_optimization(method, options):
#         return minimize(
#             to_optimize, 
#             x0=[sabr_p1.v_atm_n, sabr_p1.shift, sabr_p1.beta, sabr_p1.rho, sabr_p1.volvol], 
#             args=(strikes, pdf_p2),
#             bounds=[(0.001, .1), (0.0001, .2), (-1, 1), (-1, 1), (0.001, 10)],
#             method=method,
#             options=options
#         )
    
    
#     sabr_dist_p1 = SabrDist.from_sabr(sabr_p1)
#     sabr_dist_p2 = SabrDist.from_sabr(sabr_p2)
#     pdf_p1 = sabr_dist_p1.pdf(strikes) / sabr_dist_p1.pdf(strikes).sum()
#     pdf_p2 = sabr_dist_p2.pdf(strikes) / sabr_dist_p2.pdf(strikes).sum()
#     # Set optimization options to increase maximum number of function evaluations
#     optimization_result = run_optimization(
#         method='Nelder-Mead', 
#         options={'maxfev': 10000, 'disp': True}
#     )
#     if not optimization_result.success:
#         print(f"Optimization failed with method 'Nelder-Mead': {optimization_result.message}")
#         print("Retrying with 'Powell' method...")
#         optimization_result = run_optimization(
#             method='Powell',
#             options={'maxfev': 10000, 'disp': True}
#         )
#     print(f"Optimization result: {optimization_result}")
#     v_atm_n, shift, beta, rho, volvol = optimization_result.x

#     sabr_p1_p2 = Hagan2002NormalSABR(
#         f=sabr_p2.f, 
#         shift=shift, 
#         t=(sabr_p2.t - sabr_p1.t), 
#         v_atm_n=v_atm_n,
#         beta=beta,
#         rho=rho,
#         volvol=volvol
#     )
#     return optimization_result, sabr_p1_p2

def find_fvol(strikes, sabr_p1, sabr_p2):
    def to_optimize(x, strikes, pdf_p1, pdf_p2):
        v_atm_n, shift, beta, rho, volvol = x
        sabr = Hagan2002NormalSABR(
            f=sabr_p2.f,
            v_atm_n=v_atm_n, 
            shift=shift, 
            beta=beta, 
            rho=rho, 
            volvol=volvol,
            t=sabr_p2.t - sabr_p1.t
            )
        sabr_dist = SabrDist.from_sabr(sabr=sabr)
        vols_p1_p2 = sabr_dist.normal_vol(strikes)
        pdf = ss.norm.pdf(
            x=strikes.reshape(-1, 1) + sabr.f, 
            loc = strikes.reshape(1, -1) + sabr.f, 
            scale = vols_p1_p2.reshape(1, -1) * np.sqrt(sabr.t)
            )
        pdf_df = pd.DataFrame(
            data=pdf, 
            columns=pd.Index(strikes, name='Rate At Start'), 
            index=pd.Index(strikes, name='Rate At End')
            )
        pdf_df = pdf_df / pdf_df.sum()
        pdf_total = (pdf_df * pdf_p1).sum(axis=1)
        kl = np.sum(rel_entr(pdf_total, pdf_p2)) * 10**6
        return kl
    
    sabr_dist_p1 = SabrDist.from_sabr(sabr=sabr_p1)
    sabr_dist_p2 = SabrDist.from_sabr(sabr=sabr_p2)
    pdf_p1 = sabr_dist_p1.pdf(strikes) / sabr_dist_p1.pdf(strikes).sum()
    pdf_p2 = sabr_dist_p2.pdf(strikes) / sabr_dist_p2.pdf(strikes).sum()

    optimization_result = minimize(
            to_optimize, 
            x0=[sabr_p2.v_atm_n, sabr_p2.shift, sabr_p2.beta, sabr_p2.rho, sabr_p2.volvol], 
            args=(strikes, pdf_p1, pdf_p2),
            bounds=[(0.001, .1), (0.0001, .2), (-1, 1), (-1, 1), (0.001, 10)],
            method='Nelder-Mead',
        )    
    sabr_dist_p1_p2 = Hagan2002NormalSABR(
        f=sabr_p2.f, 
        v_atm_n=optimization_result.x[0], 
        shift=optimization_result.x[1], 
        beta=optimization_result.x[2], 
        rho=optimization_result.x[3], 
        volvol=optimization_result.x[4],
        t=sabr_p2.t - sabr_p1.t
    )
    return optimization_result, sabr_dist_p1_p2

def find_sabr_from_vols(
        strikes_values_bps: np.ndarray,
        vols_bps,
        pdf,
        f,
        T
):
    def to_optimize(x, strikes_values, vols, pdf):
        v_atm_n, shift, beta, rho, volvol = x
        sabr = Hagan2002NormalSABR(
            f=f, shift=shift, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol
        )
        # vols_computed = sabr.normal_vol(strikes_relative)
        interp_vol = get_interpolated_vols(sabr)
        vols_computed = interp_vol(strikes_values)
        error = ((vols_computed - vols) ** 2).dot(pdf) * 10**12
        return error
    
    def run_optimization(method):
        return minimize(
            to_optimize,
            x0=[0.005, .05, .5, 0.3, 0.2],
            bounds=((0.00001, .1), (0.0001, .2), (.001, .999), (-.999, 0.999), (0.00001, 3)),
            args=(strikes_values_bps / 10000, vols_bps / 10000, pdf),
            method=method,
        )

    optimization_result = run_optimization(method='Nelder-Mead')
    if not optimization_result.success:
        optimization_result = run_optimization(method='Powell')
        if not optimization_result.success:
            raise ValueError(f"Optimization failed: {optimization_result.message}")
    print(f"Optimization result find_sabr_from_vols: {optimization_result}")

    v_atm_n, shift, beta, rho, volvol = optimization_result.x
    return Hagan2002NormalSABR(
        f=f, shift=shift, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol
    )
    
# def find_sabr_from_prices(
#         strikes_relative_bps,
#         prices_rec_bps,
#         f,
#         T,
# ):
#     def to_optimize(x, strikes_relative, prices):
#         v_atm_n, shift, beta, rho, volvol = x
#         sabr = Hagan2002NormalSABR(
#             f=f, shift=shift, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol
#         )
#         vols_computed = sabr.normal_vol(strikes_relative)
#         prices_computed = compute_option_prices(
#             F=f,
#             K=strikes_relative+f,
#             vol=vols_computed,
#             T=T,
#             call_put='put'
#         )
#         error = ((prices - prices_computed) ** 2).sum() * 10**12
#         return error
    
#     def run_optimization(method):
#         return minimize(
#             to_optimize,
#             x0=[0.005, .05, .5, 0.3, 0.2],
#             bounds=((0.00001, .1), (0.0001, .2), (.001, .999), (-.999, 0.999), (0.00001, 3)),
#             args=(strikes_relative_bps / 10000, prices_rec_bps / 10000,),
#             method=method,
#         )

#     optimization_result = run_optimization(method='Nelder-Mead')
#     if not optimization_result.success:
#         optimization_result = run_optimization(method='Powell')
#         if not optimization_result.success:
#             raise ValueError(f"Optimization failed: {optimization_result.message}")
#     v_atm_n, shift, beta, rho, volvol = optimization_result.x
#     return Hagan2002NormalSABR(
#         f=f, shift=shift, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol
#     )
    
def find_sabr(
        ezbh,
        generator,
        exp,
        tail,
        method : typing.Literal['vols', 'prices'] = 'vols',
    ):
    distribution_df = ezbh.distribution_df(Generator=generator, EffExp=exp, MtyTnr=tail,).dropna()
    T = distribution_df['T'].values[0]
    strikes_relative_bps = np.array(distribution_df['strike_relative'].values) 
    strikes_bps = np.array(distribution_df['strike_float'].values) 
    prices_bps = np.array(distribution_df['price_bps'].values)
    vols_bps = np.array(distribution_df['vol'].values)
    f = distribution_df.set_index('strike_relative').loc[0, 'strike_float'] / 10000
    pdf = distribution_df['cdf'].diff().fillna(distribution_df['cdf'])
    pdf = pdf / pdf.sum()


    if method == 'vols':
        sabr = find_sabr_from_vols(
            strikes_values_bps=strikes_bps,
            vols_bps=vols_bps,
            pdf=pdf,
            f=f,
            T=T,
        )
        return sabr
    elif method == 'prices':
        raise NotImplementedError("Method 'prices' is not implemented yet.")
        # Uncomment the following lines when the method is implemented
        sabr = find_sabr_from_prices(
            strikes_values_bps=strikes_bps,
            prices_rec_bps=prices_bps,
            f=f,
            T=T,
        )
        return sabr