#!/usr/bin/python3

# import matplotlib.pyplot as plt
import os
import shutil
from decimal import *


class ReevaluationOfPriorEvidence:
    def __init__(self, causing_evidence_postion, prior_evidence_positions_to_be_updated, variance_h_1, variance_h_2):
        self.causing_evidence_postion = causing_evidence_postion
        self.prior_evidence_positions_to_be_updated = prior_evidence_positions_to_be_updated
        self.variance_h_1 = variance_h_1
        self.variance_h_2 = variance_h_2


class BayesItem:
    """
    The item we'll calculate on
    """

    def __init__(self, position, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2):
        self.position = position
        self.likelihood_h_1 = likelihood_h_1
        self.likelihood_h_2 = likelihood_h_2
        self.prior_h_1 = prior_h_1
        self.prior_h_2 = prior_h_2
        self.posterior_h_1 = 0.0
        self.posterior_h_2 = 0.0

        self.ratio_of_prior = 0.0
        self.ratio_of_likelihood = 0.0
        self.ratio = 0.0

        self.calculate_posterior()

    def calculate_posterior(self):
        """
        Bayes calculation e.g.
                             Likelihood(H_1)	    Prior(H_1)	      Likelihood(MG)	      Prior(H1)	        Likelihood(MB)	       Prior(H2)
        """

        if(((self.likelihood_h_1 * self.prior_h_1) + (self.likelihood_h_2 * self.prior_h_2)) > 0):
            self.posterior_h_1 = (self.likelihood_h_1 * self.prior_h_1) / ((self.likelihood_h_1 * self.prior_h_1) + (self.likelihood_h_2 * self.prior_h_2))
        else:
            self.posterior_h_1 = 0
        if(((self.likelihood_h_2 * self.prior_h_2) + (self.likelihood_h_1 * self.prior_h_1)) > 0):
            self.posterior_h_2 = (self.likelihood_h_2 * self.prior_h_2) / ((self.likelihood_h_2 * self.prior_h_2) + (self.likelihood_h_1 * self.prior_h_1))
        else:
            self.posterior_h_2 = 0

        self.posterior_h_1 = normalize(self.posterior_h_1)
        self.posterior_h_2 = normalize(self.posterior_h_2)

        # ratio = ratio_of_likelihood * ratio_of_prior
        self.ratio_of_likelihood = self.likelihood_h_1 / self.likelihood_h_2
        self.ratio_of_prior = self.prior_h_1 / self.prior_h_2
        self.ratio = self.ratio_of_likelihood * self.ratio_of_prior

def normalize(n):
    if n > 1:
        n = 1
    if n < 0:
        n = 0
    return n

prior_1 = .5
prior_2 = .5
likelihood_h_1 = .49
likelihood_h_2 = .51
for i in range(0,500):
    b = BayesItem(i, likelihood_h_1, prior_1, likelihood_h_2, prior_2)
    prior_1 = b.posterior_h_1
    prior_2 = b.posterior_h_2
    if i > 219:
        print("{}    {} * {} = {}  || {}::{}".format(
            i, 
            format(b.ratio_of_likelihood, ".5f"),
            format(b.ratio_of_prior, ".5f"),
            format(b.ratio, ".5f"),
            format(prior_1, ".5f"),
            format(prior_2, ".5f")
            )
        )    
