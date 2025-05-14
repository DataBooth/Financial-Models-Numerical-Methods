## Experience with Marimo (Part 2.5)

So I am using the repo [Financial-Models-Numerical-Methods](https://github.com/DataBooth/Financial-Models-Numerical-Methods) to evaluate the Marimo experience.

I started with forking the repo and cloning it to my local machine. I then created a new branch called `marimo-experience` to work on my changes.


<!-- [TODO - Cut n paste in Perplexity conversation] -->

1. BS notebook - 1.1 Black-Scholes numerical methods.ipynb

Required scp.mean() to np.mean()

        modified:   3.2 Variance Gamma model, PIDE method.ipynb
        modified:   src/FMNM/BS_pricer.py
        modified:   src/FMNM/Heston_pricer.py
        modified:   src/FMNM/Merton_pricer.py
        modified:   src/FMNM/NIG_pricer.py
        modified:   src/FMNM/VG_pricer.py

First checked that the original notebook worked. 

So the conversion worked (other than dealing with spaces in the name). The rendering of the markdown is quite beautiful.

So of the markdown may need to be fixed. I will check the original notebook and see if there are any differences. Or maybe just missed a newline or two.

e.g. The stock price on each node of a binomial tree is
$$ S_n^{(k)} = u^{k} d^{n-k} S_0 \quad \text{for} \quad k \in \{ 0, ..., n \} \quad \text{and} \quad n \in \{ 0, ..., N \}$$
where the price at $S_n^{(k)}$ is obtained starting from $S_0^{(0)}$ := $S_0$ and applying $k$ up factors and $(n-k)$ down factors. 

Not sure why lines have pre-underscore? e.g.

_S_T = np.array([S0 * u ** j * d ** (N_1 - j) for j in range(N_1 + 1)])

Adjusting the display width under settings is helpful.

Redefinition / import of 

import marimo as mo

at the end of the notebook?? Removed and included at top od the notebook - seemed to fix the issue.

Nice being able to run with the markdown version of the notebook (see [justfile](justfile) for commands).


Tried using the Marimo extension for VS Code and it seemed really slow and had issues.


Also seeing this error - is it real?

/Users/mjboothaus/code/github/databooth/Financial-Models-Numerical-Methods/src/FMNM/Kalman_filter.py:316: SyntaxWarning: invalid escape sequence '\p'
  label="Kalman Std Dev: $\pm 1 \sigma$",
/Users/mjboothaus/code/github/databooth/Financial-Models-Numerical-Methods/src/FMNM/Merton_pricer.py:33: SyntaxWarning: invalid escape sequence '\i'
  + \int[ V(x+y) nu(dy) ] -(r+lam)V
/Users/mjboothaus/code/github/databooth/Financial-Models-Numerical-Methods/src/FMNM/NIG_pricer.py:33: SyntaxWarning: invalid escape sequence '\i'
  + \int[ V(x+y) nu(dy) ] -(r+lam)V
/Users/mjboothaus/code/github/databooth/Financial-Models-Numerical-Methods/src/FMNM/VG_pricer.py:34: SyntaxWarning: invalid escape sequence '\i'
  + \int[ V(x+y) nu(dy) ] -(r+lam)V
  

  Seems had to "wrap" these as raw strings with r-prefix and then ok.


