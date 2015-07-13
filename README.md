# scde-scratch
Expectation maximisation for concomitant variable NB models in python

This python script implements a simplified version of 'Bayesian approach to single-cell differential expression analysis'
 (Nature Methods 2014), but from scratch (rather than using the FlexMix R package) and in python.

 It essentially performs EM inference on a poisson-negative binomial model where the mixing distribution is dependent on each genes average expression (known as concomitant variable model or mixture of experts regression).