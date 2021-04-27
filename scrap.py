#%% Misc Array Testing
import numpy as np
a = np.array([1,2,3])

b = 1 + 2j

#%% Printing monomials
import numpy as np
from sympy import symbols
from sympy.polys.monomials import itermonomials, monomial_count
from sympy.polys.orderings import monomial_key

x_str = ""
for i in range(2):
    x_str += 'x_' + str(i) + ', '
x_syms = symbols(x_str)
M = itermonomials(x_syms, 5)
sortedM = sorted(M, key=monomial_key('grlex', np.flip(x_syms)))
print(sortedM)

# %%
