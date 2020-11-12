#!/usr/bin/python3

# import matplotlib.pyplot as plt
from decimal import *
import random
from copy import deepcopy


class Config:
    """
    How to perform this calc
    """

    def __init__(self, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations):
        self.likelihood_h_1 = likelihood_h_1
        self.likelihood_h_2 = likelihood_h_2
        self.prior_h_1 = prior_h_1
        self.prior_h_2 = prior_h_2
        self.iterations = iterations
        self.iteration_to_start_reevaluation = 0
        self.reevaluation_likelihood_adjustment_low = 0
        self.reevaluation_likelihood_adjustment_high = 0
        self.evidence_reevaluations = []


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


def lookback(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations, iteration_to_update, variance_low, variance_high, lookback_reevaluations, is_debug=False):
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
    original_likelihood_h_1 = likelihood_h_1
    original_likelihood_h_2 = likelihood_h_2
    items = []

    for i in range(iterations):
        if i >= iteration_to_update:  # where the potential for reevaluation begins
            if i in lookback_reevaluations:
                lookback_items = []
                for update_count, item in enumerate(items):
                    if update_count == 0:
                        prior_h_1 = item.prior_h_1
                        prior_h_2 = item.prior_h_2

                    item_original_likelihood_h_1 = item.likelihood_h_1
                    item_original_likelihood_h_2 = item.likelihood_h_2

                    x = ""
                    # is this a reevalution E?
                    if update_count in lookback_reevaluations:
                        change = random.uniform(variance_low, variance_high)
                        likelihood_h_1 = normalize(item.likelihood_h_1 + change)
                        likelihood_h_2 = normalize(item.likelihood_h_2 + (1-abs(change)))
                        x = "reevaluation likelihood " + str(likelihood_h_1)
                    else:
                        likelihood_h_1 = item_original_likelihood_h_1
                        likelihood_h_2 = item_original_likelihood_h_2
                        x = "straight " + str(likelihood_h_1)

                    lookback_item = BayesItem(update_count, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
                    lookback_items.append(lookback_item)

                    # set item's priors to iteration's prior
                    prior_h_1 = lookback_item.posterior_h_1
                    prior_h_2 = lookback_item.posterior_h_2
                    if is_debug:
                        print(str(i) + " " + str(update_count) + " " + x)

            else:  # evaluate as normal
                likelihood_h_1 = original_likelihood_h_1
                likelihood_h_2 = original_likelihood_h_2

            # this is the summary lookback calculation
            b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
            # we don't add this to a list, we just need it's output
            prior_h_1 = b.posterior_h_1
            prior_h_2 = b.posterior_h_2

        if is_debug:
            print(str(i) + " " + str(prior_h_1))

        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)
        prior_h_1 = b.posterior_h_1
        prior_h_2 = b.posterior_h_2

    return items


def normalize(n):
    if n > 1:
        n = 1
    if n < 0:
        n = 0
    return n


getcontext().prec = 5

ITERATIONS = 25
ITERATIONTOSTARTREEVALUATION = 7
# for reevaluation, this array needs to contain the start iteration number
LOOKBACKREEVALUATIONS = [1, 3, 5, 7, 8, 9, 15, 16]
REEVALUATIONLOW = -.3
REEVALUATIONHIGH = -.3
LIKELIHOOD_H_1 = .6
LIKELIHOOD_H_2 = .4
PRIOR_H_1 = .5
PRIOR_H_2 = .5


config = Config(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS)
config.iteration_to_start_reevaluation = ITERATIONTOSTARTREEVALUATION
config.reevaluation_likelihood_adjustment_low = REEVALUATIONLOW
config.reevaluation_likelihood_adjustment_high = REEVALUATIONHIGH
config.evidence_reevaluations = LOOKBACKREEVALUATIONS

results1 = bayes(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS)

results2 = single_update(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                         ITERATIONTOSTARTREEVALUATION, REEVALUATIONLOW, REEVALUATIONHIGH)

results3 = iterative(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                     ITERATIONTOSTARTREEVALUATION, REEVALUATIONLOW, REEVALUATIONHIGH)

results4 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                    ITERATIONTOSTARTREEVALUATION, REEVALUATIONLOW, REEVALUATIONHIGH, [2, 4, 7])

results5 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                    ITERATIONTOSTARTREEVALUATION, REEVALUATIONLOW, REEVALUATIONHIGH, [5, 6, 7], True)


# for i, result in enumerate(results3):
# print(str(i) + "," + "{:5f}".format(result.Posterior_H_1) +
#       "," + "{:5f}".format(result.Prior_H_1))
# print("{:5f}".format(result.Posterior_H_1))


print("E  ,Bayes,Single,Iterative,LookbackSingle,LookbackIterative")
for i in range(ITERATIONS):
    print("{0:03d}".format(i) + "," +
          "{:5f}".format(results1[i].posterior_h_1) + "," +
          "{:5f}".format(results2[i].posterior_h_1) + "," +
          "{:5f}".format(results3[i].posterior_h_1) + "," +
          "{:5f}".format(results4[i].posterior_h_1) + "," +
          "{:5f}".format(results5[i].posterior_h_1))
