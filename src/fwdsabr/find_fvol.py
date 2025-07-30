import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
from scipy.special import rel_entr
from pysabr import Hagan2002NormalSABR

class SabrDist(ss.rv_continuous):
    def __init__(self, f, shift, t, v_atm_n, beta, rho, volvol):
        super().__init__(name='SABR', a=-shift+1e-6, )
        self.sabr = Hagan2002NormalSABR(f=f, shift=shift, t=t, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol)
    
    def _cdf(self, x): # type: ignore
        vols = self.sabr.normal_vol(x)
        return ss.norm.cdf(x, loc = self.sabr.f, scale = vols * np.sqrt(self.sabr.t))
    
    @classmethod
    def from_sabr(cls, sabr):
        return cls(f=sabr.f, shift=sabr.shift, t=sabr.t, v_atm_n=sabr.v_atm_n, beta=sabr.beta, rho=sabr.rho, volvol=sabr.volvol)

def get_total_pdf(sabr_p1, sabr_p1_p2, strikes):
    sabr_dist_p1 = SabrDist.from_sabr(sabr_p1)
    sabr_dist_p1_p2 = SabrDist.from_sabr(sabr_p1_p2)
    pdf_p1 = sabr_dist_p1.pdf(strikes) / sabr_dist_p1.pdf(strikes).sum()
    
    vols_p1_p2 = sabr_p1_p2.normal_vol(strikes)
    pdf = ss.norm.pdf(strikes.reshape(-1, 1), loc = strikes.reshape(1, -1), scale = vols_p1_p2.reshape(1, -1) * np.sqrt(.5))
    
    pdf = pdf / pdf.sum()
    pdf_total = (pdf * pdf_p1).sum(axis=1)
    return pdf_total

def find_fvol(sabr_p1, sabr_p2, strikes):
    def to_optimize(x, strikes, pdf_p1, pdf_p2):
        v_atm_n, beta, rho, volvol = x
        sabr_p1_p2 = Hagan2002NormalSABR(f=sabr_p1.f, shift=sabr_p1.shift, t=sabr_p1.t, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol)
        vols_p1_p2 = sabr_p1_p2.normal_vol(strikes)
        pdf = ss.norm.pdf(strikes.reshape(-1, 1), loc = strikes.reshape(1, -1), scale = vols_p1_p2.reshape(1, -1) * np.sqrt(.5))
        pdf = pdf / pdf.sum(axis = 0)
        pdf_total = (pdf * pdf_p1).sum(axis=1)
        kl = np.sum(rel_entr(pdf_total, pdf_p2)) * 10**6
        return kl
    
    sabr_dist_p1 = SabrDist.from_sabr(sabr_p1)
    sabr_dist_p2 = SabrDist.from_sabr(sabr_p2)
    pdf_p1 = sabr_dist_p1.pdf(strikes) / sabr_dist_p1.pdf(strikes).sum()
    pdf_p2 = sabr_dist_p2.pdf(strikes) / sabr_dist_p2.pdf(strikes).sum()
    
    optimization_result = minimize(
        to_optimize, 
        x0=[sabr_p1.v_atm_n, sabr_p1.beta, sabr_p1.rho, sabr_p1.volvol], 
        args=(strikes, pdf_p1, pdf_p2),
        bounds=[(0.001, .1), (-1, 1), (-1, 1), (0.001, 10)],
        method='Nelder-Mead',  # Nelder-Mead is often a good choice for non-smooth functions
        # method='Powell',
        # method='L-BFGS-B',  # More efficient for large problems
    )
    v_atm_n, beta, rho, volvol = optimization_result.x

    sabr_p1_p2 = Hagan2002NormalSABR(
        f=sabr_p1.f, 
        shift=sabr_p1.shift, 
        t=sabr_p1.t, 
        v_atm_n=v_atm_n,
        beta=beta,
        rho=rho,
        volvol=volvol
    )
    return optimization_result, sabr_p1_p2

    