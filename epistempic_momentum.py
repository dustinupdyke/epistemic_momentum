#!/usr/bin/python3

# import matplotlib.pyplot as plt
from decimal import *
import random
from copy import deepcopy


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


def single_update(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations, iteration_to_update, variance_h_1, variance_h_2):
    """
    one belief update, then return to normal likelihoods
    """
    original_likelihood_h_1 = likelihood_h_1
    original_likelihood_h_2 = likelihood_h_2
    items = []
    for i in range(iterations):
        if i == iteration_to_update:
            # Change in belief changes the likelihood, not the prior or posterior (that would be "magic")
            likelihood_h_1 = normalize(likelihood_h_1 + variance_h_1)
            likelihood_h_2 = normalize(likelihood_h_2 + variance_h_2)
        else:
            likelihood_h_1 = original_likelihood_h_1
            likelihood_h_2 = original_likelihood_h_2

        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)
        prior_h_1 = b.posterior_h_1
        prior_h_2 = b.posterior_h_2
    return items


def iterative(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations, iteration_to_update, variance_h_1, variance_h_2):
    """
    Continue to update in the same one fashion until convergence
    So for each iteration above our belief change, we recalculate likelihoods
    """
    items = []
    for i in range(iterations):
        if i >= iteration_to_update:

            has_processed = False
            if(i > 1500):
                likelihood_h_1 = .51
                likelihood_h_2 = .49
                has_processed = True
            if(i > 1000 and not has_processed):
                likelihood_h_1 = .48
                likelihood_h_2 = .52
                has_processed = True

            if(i > 500 and not has_processed):
                likelihood_h_1 = .49
                likelihood_h_2 = .51
                has_processed = True

            if(not has_processed):
                likelihood_h_1 = .51
                likelihood_h_2 = .49

            # likelihood_h_1 = normalize(likelihood_h_1 + variance_h_1)
            # likelihood_h_2 = normalize(likelihood_h_2 + variance_h_2)

        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)
        prior_h_1 = b.posterior_h_1
        prior_h_2 = b.posterior_h_2
    return items


def lookback(likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2, iterations, reevaluations, is_debug=False):
    """
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
    if is_debug:
        print("i,iter,likelihood_h_1,prior_h_1,b.posterior_h_1,prior_h_2,posterior_h_2")

    items = []

    original_likelihood_h_1 = likelihood_h_1
    original_likelihood_h_2 = likelihood_h_2

    for i in range(iterations):
        reevaluation = None
        for r in reevaluations:
            # print(str(r.causing_evidence_postion) + "," + str(r.prior_evidence_positions_to_be_updated))
            if r.causing_evidence_postion == i:
                reevaluation = r
                break

        # is this a new E that causes reevaluation?
        if reevaluation:
            # print("reevaluation at " + str(i))
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
                if update_count in r.prior_evidence_positions_to_be_updated:
                    likelihood_h_1 = normalize(item.likelihood_h_1 + reevaluation.variance_h_1)
                    likelihood_h_2 = normalize(item.likelihood_h_2 + reevaluation.variance_h_2)
                else:
                    likelihood_h_1 = item_original_likelihood_h_1
                    likelihood_h_2 = item_original_likelihood_h_2

                lookback_item = BayesItem(update_count, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
                lookback_items.append(lookback_item)

                if is_debug:
                    print("{0:03d}".format(i) + "," +
                          "{0:03d}".format(update_count) + "," + str(likelihood_h_1) + "," +
                          "{:5f}".format(prior_h_1) + "," + "{:5f}".format(lookback_item.posterior_h_1) + "," +
                          "{:5f}".format(prior_h_2) + "," + "{:5f}".format(lookback_item.posterior_h_2) + "," + str(r.causing_evidence_postion) + "," + str(r.prior_evidence_positions_to_be_updated))

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

ITERATIONS = 2000
LIKELIHOOD_H_1 = .51
LIKELIHOOD_H_2 = .49
PRIOR_H_1 = .5
PRIOR_H_2 = .5
ITERATIONTOSTARTREEVALUATION = 10
REEVALUATION_H_1 = -.05
REEVALUATION_H_2 = .05

DEBUG = False

# if (DEBUG):
#     print("i,update_i,likelihood_h_1,lookback_item.prior_h_1,lookback_item.posterior_h_1,lookback_item.prior_h_2,lookback_item.posterior_h_2")

results1 = bayes(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS)

results2 = single_update(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                         ITERATIONTOSTARTREEVALUATION, REEVALUATION_H_1, REEVALUATION_H_2)

results3 = iterative(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS,
                     ITERATIONTOSTARTREEVALUATION, REEVALUATION_H_1, REEVALUATION_H_2)

reevaluations = []
reevaluations.append(ReevaluationOfPriorEvidence(ITERATIONTOSTARTREEVALUATION, [2], REEVALUATION_H_1, REEVALUATION_H_2))

results4 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
evidence_to_reeval = []
for i in range(ITERATIONS):
    if i >= ITERATIONTOSTARTREEVALUATION:
        evidence_to_reeval.append(i)
        l = evidence_to_reeval[:]

        has_processed = False

        if(i > 1500):
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, .51, .49))
            has_processed = True

        if(i > 1000 and not has_processed):
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, .48, .52))
            has_processed = True

        if(i > 500 and not has_processed):
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, .49, .51))
            has_processed = True

        if(not has_processed):
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, .51, .49))


if DEBUG:
    for r in reevaluations:
        print(r.causing_evidence_postion, r.prior_evidence_positions_to_be_updated, r.variance_h_1, r.variance_h_2)

results5 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

# exit(66)

reevaluations = []
evidence_to_reeval = []
for i in range(ITERATIONS):
    if i >= ITERATIONTOSTARTREEVALUATION:
        if(i % 10 == 0):
            for i in range(ITERATIONS):
                if i >= ITERATIONTOSTARTREEVALUATION:
                    evidence_to_reeval.append(i)
                    l = evidence_to_reeval[:]

                    has_processed = False

                    if(i > 1500):
                        reevaluations.append(ReevaluationOfPriorEvidence(i, l, .51, .49))
                        has_processed = True

                    if(i > 1000 and not has_processed):
                        reevaluations.append(ReevaluationOfPriorEvidence(i, l, .48, .52))
                        has_processed = True

                    if(i > 500 and not has_processed):
                        reevaluations.append(ReevaluationOfPriorEvidence(i, l, .49, .51))
                        has_processed = True

                    if(not has_processed):
                        reevaluations.append(ReevaluationOfPriorEvidence(i, l, .51, .49))


results6 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
evidence_to_reeval = []
for i in range(ITERATIONS):
    if i >= ITERATIONTOSTARTREEVALUATION:
        if(i % 5 == 0):
            evidence_to_reeval.append(i-1)
            l = evidence_to_reeval[:]
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, REEVALUATION_H_1, REEVALUATION_H_2))

results7 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
evidence_to_reeval = []
for i in range(ITERATIONS):
    if i >= ITERATIONTOSTARTREEVALUATION:
        if(i % 3 == 3):
            evidence_to_reeval.append(i-1)
            l = evidence_to_reeval[:]
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, REEVALUATION_H_1, REEVALUATION_H_2))

results8 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
evidence_to_reeval = []
for i in range(ITERATIONS):
    if i >= ITERATIONTOSTARTREEVALUATION:
        if(i % 2 == 0):
            evidence_to_reeval.append(i-1)
            l = evidence_to_reeval[:]
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, REEVALUATION_H_1, REEVALUATION_H_2))

results9 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)

reevaluations = []
evidence_to_reeval = []
for i in range(ITERATIONS):
    if i >= ITERATIONTOSTARTREEVALUATION:
        if(i % 2 != 0):
            evidence_to_reeval.append(i-1)
            l = evidence_to_reeval[:]
            reevaluations.append(ReevaluationOfPriorEvidence(i, l, REEVALUATION_H_1, REEVALUATION_H_2))

results10 = lookback(LIKELIHOOD_H_1, PRIOR_H_1, LIKELIHOOD_H_2, PRIOR_H_2, ITERATIONS, reevaluations, DEBUG)


with open("output.txt", "w") as f:
    # f.write("E  ,Bayes,Single,Iterative,LookbackSingle,LookbackIterative\n")
    for i in range(ITERATIONS):
        f.write("{0:03d}".format(i) + "," +
                "{:5f}".format(results1[i].posterior_h_1) + "," +
                "{:5f}".format(results2[i].posterior_h_1) + "," +
                "{:5f}".format(results3[i].posterior_h_1) + "," +
                "{:5f}".format(results4[i].posterior_h_1) + "," +
                "{:5f}".format(results5[i].posterior_h_1) + "," +
                "{:5f}".format(results6[i].posterior_h_1) + "," +
                "{:5f}".format(results7[i].posterior_h_1) + "," +
                "{:5f}".format(results8[i].posterior_h_1) + "," +
                "{:5f}".format(results9[i].posterior_h_1) + "," +
                "{:5f}".format(results10[i].posterior_h_1) + "\n")

print("Output file written")
exit(0)

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
          "{:5f}".format(results9[i].posterior_h_1) + "," +
          "{:5f}".format(results10[i].posterior_h_1))
