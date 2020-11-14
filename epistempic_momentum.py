#!/usr/bin/python3

# import matplotlib.pyplot as plt
from decimal import *
import random
from copy import deepcopy


class ReevaluationOfPriorEvidence:
    def __init__(self, causing_evidence_postion, priors_to_be_updated, belief_change_low, belief_change_high):
        self.causing_evidence_postion = causing_evidence_postion
        self.priors_to_be_updated = priors_to_be_updated
        self.belief_change_low = belief_change_low
        self.belief_change_high = belief_change_high


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
        self.calculate_posterior()

    def calculate_posterior(self):
        """
        Bayes calculation e.g.
                             Likelihood(H_1)	    Prior(H_1)	      Likelihood(MG)	      Prior(H1)	        Likelihood(MB)	       Prior(H2)
        """
        self.posterior_h_1 = (self.likelihood_h_1 * self.prior_h_1) / ((self.likelihood_h_1 * self.prior_h_1) + (self.likelihood_h_2 * self.prior_h_2))
        self.posterior_h_2 = (self.likelihood_h_2 * self.prior_h_2) / ((self.likelihood_h_2 * self.prior_h_2) + (self.likelihood_h_1 * self.prior_h_1))

        self.posterior_h_1 = normalize(self.posterior_h_1)
        self.posterior_h_2 = normalize(self.posterior_h_2)


def bayes(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations):
    """
    straight iterative bayes calculation, where priors become the previous posterior
    """
    items = []
    for i in range(iterations):
        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)
        # priors for the next are this iterations posterior
        prior_h_1 = b.posterior_h_1
        prior_h_2 = b.posterior_h_2
    return items


def single_update(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations, iteration_to_update, variance_low, variance_high):
    """
    one belief update, then return to normal likelihoods
    """
    original_likelihood_h_1 = likelihood_h_1
    original_likelihood_h_2 = likelihood_h_2
    items = []
    for i in range(iterations):
        if i == iteration_to_update:
            # Change in belief changes the likelihood, not the prior or posterior (that would be "magic")
            change = random.uniform(variance_low, variance_high)
            likelihood_h_1 = normalize(likelihood_h_1 + change)
            likelihood_h_2 = normalize(likelihood_h_2 + (1-abs(change)))
        else:
            likelihood_h_1 = original_likelihood_h_1
            likelihood_h_2 = original_likelihood_h_2

        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)
        prior_h_1 = b.posterior_h_1
        prior_h_2 = b.posterior_h_2
    return items


def iterative(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations, iteration_to_update, variance_low, variance_high):
    """
    Continue to update in the same one fashion until convergence
    So for each iteration above our belief change, we recalculate likelihoods
    """
    items = []
    for i in range(iterations):
        if i >= iteration_to_update:
            change = random.uniform(variance_low, variance_high)
            likelihood_h_1 = normalize(likelihood_h_1 + change)
            likelihood_h_2 = normalize(likelihood_h_2 + (1-abs(change)))

        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)
        prior_h_1 = b.posterior_h_1
        prior_h_2 = b.posterior_h_2
    return items


def lookback(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations, reevaluations, is_debug=False):
    """
    My use of E and F was just to distinguish between the evidence received before
    the “realization moment” and the evidence received after it (respectively).
    So P*(E | H_1) is the likelihood/probability that I would see data like
    E if H_1 were true. That’s not the likelihood I used when I first observed E,
    of course, since I first saw E before I realized that I should use a different
    likelihood function. But if I do retrospective re-evaluation, then I will need
    to compute it, since I’m thinking “here’s what I should have believed back then.”

    In terms of convergence to 1 vs. to 0, the individual likelihood actually doesn’t
    matter. What matters is the **ratio** of the likelihoods for H_1 and for H_2.
    In general, if P(E | H_1) > P(E | H_2),
    then the Bayesian will (given infinite amounts of E-data) converge to P(H_1) = 1;
    and obviously, P(H_1) will converge to 0 if the inequality is reversed.
    So when you change the likelihoods at the “realization moment,”
    does that lead to a reversal of the inequality?

    --------

    For now, I would focus on the simple case where the re-evaluation applies to all prior evidence uniformly.
    That is, if we have seen evidence E (bolded to indicate that it is a set of evidence, not just one piece),
    then we have posteriors:

    P(H_1 | E) = P(E | H_1) P(H_1) / P(E)
    P(H_2 | E) = P(E | H_2) P(H_2) / P(E)

    If the likelihoods change to P*, then when we get new evidence F, we could:
    (a) use the above posteriors as our priors (i.e., forget the past evidence) then we have
    (dropping the denominators for simplicity):
    —  P*(F | H_1) P(H_1 | E) = P*(F | H_1) P(E | H_1) P(H_1)
    —  P*(F | H_2) P(H_2 | E) = P*(F | H_2) P(E | H_2) P(H_2)
    [these will be different than if we continued using the original likelihoods,
    but they assume that we don’t change any beliefs at the "moment of insight”
    that leads to the likelihood changes]

    (b) go back and revisit the evidence E to rethink our beliefs at the moment of insight, in which case
    the numerators are:
    —  P*(F | H_1) P*(E | H_1) P(H_1)
    —  P*(F | H_2) P*(E | H_2) P(H_2)
    [that is, we compute the beliefs after E + F that we would have had if we had been using the P*s from
    the very beginning]

    1
    11
    111
    1111
    11111
    111111
    1111111   -- reevaluation begins
    11r1r1r12
    333333333

    """
    items = []

    original_likelihood_h_1 = likelihood_h_1
    original_likelihood_h_2 = likelihood_h_2

    for i in range(iterations):
        reevaluation = None
        for r in reevaluations:
            if r.causing_evidence_postion == i:
                reevaluation = r

        # is this a new E that causes reevaluation?
        if reevaluation:
            lookback_items = []
            # go back and reevaluate all items to this point
            for update_count, item in enumerate(items):
                # previous E originals
                item_original_likelihood_h_1 = item.likelihood_h_1
                item_original_likelihood_h_2 = item.likelihood_h_2
                if update_count == 0:
                    prior_h_1 = item.prior_h_1
                    prior_h_2 = item.prior_h_2

                # is this a previous E to reevaluate?
                if update_count in r.priors_to_be_updated:
                    change = random.uniform(r.belief_change_low, r.belief_change_high)
                    likelihood_h_1 = normalize(item.likelihood_h_1 + change)
                    likelihood_h_2 = normalize(item.likelihood_h_2 + (1-abs(change)))
                else:
                    likelihood_h_1 = item_original_likelihood_h_1
                    likelihood_h_2 = item_original_likelihood_h_2

                lookback_item = BayesItem(update_count, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
                lookback_items.append(lookback_item)

                if is_debug:
                    print("{0:03d}".format(i) + "," +
                          "{0:03d}".format(update_count) + "," + str(likelihood_h_1) + "," +
                          "{:5f}".format(prior_h_1) + "," + "{:5f}".format(lookback_item.posterior_h_1) + "," +
                          "{:5f}".format(prior_h_2) + "," + "{:5f}".format(lookback_item.posterior_h_2))

                prior_h_1 = lookback_item.posterior_h_1
                prior_h_2 = lookback_item.posterior_h_2

        likelihood_h_1 = original_likelihood_h_1
        likelihood_h_2 = original_likelihood_h_2

        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)

        if is_debug:
            print("{0:03d}".format(i) + "," +
                  "000," + str(likelihood_h_1) + "," +
                  "{:5f}".format(prior_h_1) + "," + "{:5f}".format(b.posterior_h_1) + "," +
                  "{:5f}".format(prior_h_2) + "," + "{:5f}".format(b.posterior_h_2))

        prior_h_1 = b.posterior_h_1
        prior_h_2 = b.posterior_h_2

    return items


def normalize(n):
    if n > 1:
        n = 1
    if n < 0:
        n = 0
    return n


print("""
 ________  ____    ____  
|_   __  ||_   \  /   _| 
  | |_ \_|  |   \/   |   
  |  _| _   | |\  /| |   
 _| |__/ | _| |_\/_| |_  
|________||_____||_____| 
                         """)

getcontext().prec = 5

ITERATIONS = 25
LIKELIHOOD_H_1 = .6
LIKELIHOOD_H_2 = .4
PRIOR_H_1 = .5
PRIOR_H_2 = .5
ITERATIONTOSTARTREEVALUATION = 7
REEVALUATIONLOW = -.5
REEVALUATIONHIGH = -.5

DEBUG = False

print("i,update_i,likelihood_h_1,lookback_item.prior_h_1,lookback_item.posterior_h_1,lookback_item.prior_h_2,lookback_item.posterior_h_2")

results1 = bayes(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS)

results2 = single_update(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                         ITERATIONTOSTARTREEVALUATION, REEVALUATIONLOW, REEVALUATIONHIGH)

results3 = iterative(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                     ITERATIONTOSTARTREEVALUATION, REEVALUATIONLOW, REEVALUATIONHIGH)

reevaluations = []
reevaluations.append(ReevaluationOfPriorEvidence(ITERATIONTOSTARTREEVALUATION, [1], REEVALUATIONLOW, REEVALUATIONHIGH))

results4 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
reevaluations.append(ReevaluationOfPriorEvidence(ITERATIONTOSTARTREEVALUATION, [2], REEVALUATIONLOW, REEVALUATIONHIGH))

results5 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
reevaluations.append(ReevaluationOfPriorEvidence(ITERATIONTOSTARTREEVALUATION, [3], REEVALUATIONLOW, REEVALUATIONHIGH))

results6 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
reevaluations.append(ReevaluationOfPriorEvidence(ITERATIONTOSTARTREEVALUATION, [4], REEVALUATIONLOW, REEVALUATIONHIGH))

results7 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
reevaluations.append(ReevaluationOfPriorEvidence(ITERATIONTOSTARTREEVALUATION, [5], REEVALUATIONLOW, REEVALUATIONHIGH))

results8 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
reevaluations.append(ReevaluationOfPriorEvidence(ITERATIONTOSTARTREEVALUATION, [6], REEVALUATIONLOW, REEVALUATIONHIGH))

results9 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

print("")
print("E  ,Bayes,Single,Iterative,LookbackSingle,LookbackIterative")
for i in range(ITERATIONS):
    print("{0:03d}".format(i) + "," +
          "{:5f}".format(results1[i].posterior_h_1) + "," +
          "{:5f}".format(results2[i].posterior_h_1) + "," +
          "{:5f}".format(results3[i].posterior_h_1) + "," +
          "{:5f}".format(results4[i].posterior_h_1) + "," +
          "{:5f}".format(results5[i].posterior_h_1) + "," +
          "{:5f}".format(results6[i].posterior_h_1) + "," +
          "{:5f}".format(results7[i].posterior_h_1) + "," +
          "{:5f}".format(results8[i].posterior_h_1) + "," +
          "{:5f}".format(results9[i].posterior_h_1))
