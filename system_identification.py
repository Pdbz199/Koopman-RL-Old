#%%
from base import *
from estimate_L import *
from estimate_drift_and_variance import EstimateDandV

L_gedmd = gedmd()
gedmd_identifier = EstimateDandV(L_gedmd)

L_sindy = SINDy()
SINDy_identifier = EstimateDandV(L_sindy)

L_rrr = rrr()
RRR_identifier = EstimateDandV(L_rrr)