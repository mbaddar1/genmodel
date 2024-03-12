"""
Why B-Splines for TT GenModel
----------------------
Tensor-Train Density Estimation
https://arxiv.org/pdf/2108.00089.pdf
Quote: (Sec. 5) Experiments, first paragraph
"In all the Experiment basis function set consists of B-splines of degree 2 with knots uniformly distributed over the
support of the considered distributions. The support is known precisely for the simulated examples as we know exactly
the target distribution, and corresponding lower and upper bounds are extracted from all given samples for the unknown
distributions of the real-world datasets."

B-Splines Resources
--------------------

B-spline Basis Functions: Definition
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html

B-spline Basis Functions: Computation Examples
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-ex-1.html

A short introduction to splines in least squares regression analysis
https://epub.uni-regensburg.de/27968/1/DP472_Kagerer_introduction_splines.pdf

B-Spline Network
https://www.duo.uio.no/bitstream/handle/10852/61162/thesisDouzette.pdf?sequence=1&isAllowed=y
Code https://github.com/AndreDouzette/BsplineNetworks

Python/Numpy implementation of Bspline basis functions
https://johnfoster.pge.utexas.edu/blog/posts/pythonnumpy-implementation-of-bspline-basis-functions/
"""