#!/usr/bin/python3

import os
import shutil
from decimal import *
import random
from datetime import datetime

"""
Industrial control systems, computer software and certainly sophisticated malware are subject 
to many different input values, and so we attempt to simulate two competing pieces of evidence 
for the same hypothesis H_1, one of high probability and one of low.
"""


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


def iterative(stuxnet=False, lookback=9999, sere = False):
    """
    Continue to update in the same one fashion until convergence
    So for each iteration above our belief change, we recalculate likelihoods
    """
    iterations = 2000
    chance_likelihood_increases = 50  # 50%
    chance_of_reset = 5

    likelihood_h_1 = .50
    prior_h_1 = .50
    likelihood_h_2 = .50
    prior_h_2 = .50

    items = []
    resets = []
    print("")
    for i in range(iterations):
        print("Processing iterative" + str(i), end='\r')

        """
        P(f) = c(x(t)) * .001 every t there is a chance of wear
        every x(t) % of manual check, resets P(f) to 0
        """
        has_processed = False
        likelihood_increases = percent_of_the_time(chance_likelihood_increases)
        if likelihood_increases:
            likelihood_h_1 = likelihood_h_1 + .000550
            likelihood_h_2 = 1 - likelihood_h_1

        if(percent_of_the_time(chance_of_reset)):  # reset via manual investigation
            likelihood_h_1 = .495
            likelihood_h_2 = .505
            resets.append(i)

        # stuxnet intervention
        if(stuxnet and likelihood_h_1 > .51 and likelihood_increases):
            if sere:
                temp_likelihood_h_1 = likelihood_h_1 - .006
                temp_likelihood_h_2 = 1 - likelihood_h_1
                if temp_likelihood_h_1 - likelihood_h_1 < 0:
                    likelihood_h_1 = likelihood_h_1 - .001
                else:
                    likelihood_h_1 = likelihood_h_1 - .006
                likelihood_h_2 = 1 - likelihood_h_1
            else:
                likelihood_h_1 = likelihood_h_1 - .011
                likelihood_h_2 = 1 - likelihood_h_1

        original_likelihood_h_1 = likelihood_h_1
        original_likelihood_h_2 = likelihood_h_2

        if(i >= lookback):
            reevaluation = ReevaluationOfPriorEvidence(lookback, [], .501, .499)

            for update_count, item in enumerate(items):
                print("Reevaluating {}:{}...".format(i, update_count), end='\r')
                # previous E originals
                item_original_likelihood_h_1 = item.likelihood_h_1
                item_original_likelihood_h_2 = item.likelihood_h_2
                if update_count == 0:
                    prior_h_1 = item.prior_h_1
                    prior_h_2 = item.prior_h_2

                likelihood_h_1 = item_original_likelihood_h_1
                likelihood_h_2 = item_original_likelihood_h_2

                # is this a previous E to reevaluate?
                likelihood_h_1 = normalize(reevaluation.variance_h_1)
                likelihood_h_2 = normalize(reevaluation.variance_h_2)

                lookback_item = BayesItem(update_count, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)

                # if is_debug:
                #     DEBUG_RAW.append("{0:03d}".format(i) + "," +
                #                      "{0:03d}".format(update_count) + "," +
                #                      str(likelihood_h_1) + "," +
                #                      "{:5f}".format(prior_h_1) + "," +
                #                      "{:5f}".format(lookback_item.posterior_h_1) + "," +
                #                      "{:5f}".format(prior_h_2) + "," +
                #                      "{:5f}".format(lookback_item.posterior_h_2) + "," +
                #                      str(reevaluation.causing_evidence_postion) + "," +
                #                      str(reevaluation.prior_evidence_positions_to_be_updated))

                prior_h_1 = lookback_item.posterior_h_1
                prior_h_2 = lookback_item.posterior_h_2

        likelihood_h_1 = original_likelihood_h_1
        likelihood_h_2 = original_likelihood_h_2

        # now how to incorporate the lookback with resets and such

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


def percent():
    return random.randint(1, 100)


def percent_of_the_time(p: int):
    x = percent()
    return x < p

def main():
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
    PRIOR_H_1 = .5
    PRIOR_H_2 = .5

    DEBUG = True
    DEBUG_RAW = []
    DEBUG_REEVALS = []

    # i want to build the array of E and their likelihoods here, and pass them into a simple function of calculation....
    results1 = iterative()
    results2 = iterative()
    results3 = iterative()
    results4 = iterative()
    results5 = iterative()
    results6 = iterative()
    results7 = iterative()
    results8 = iterative()
    results9 = iterative(False, 1000)
    results10 = iterative(False, 1000, False)
    results11 = iterative(True, 9999, False) #stuxnet=False, lookback=9999, sere = False

    if not os.path.isdir("output"):
        os.makedirs("output")

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    with open("output/stuxnet_reevaluation_results_{}.txt".format(date), "w") as f:
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
                    "{:5f}".format(results10[i].posterior_h_1) + "," +
                    "{:5f}".format(results11[i].posterior_h_1) + "\n")

    with open("output/stuxnet_reevaluation_output_reevals_{}.txt".format(date), "w") as f:
        f.write("causing_evidence_postion, prior_evidence_positions_to_be_updated, variance_h_1, variance_h_2")
        for r in DEBUG_REEVALS:
            f.write(str(r.causing_evidence_postion) + "," + str(r.prior_evidence_positions_to_be_updated) + "," + str(r.variance_h_1) + "," + str(r.variance_h_2) + "\n")

    with open("output/stuxnet_reevaluation_output_raw_debug_{}.txt".format(date), "w") as f:
        f.write("i,iter,likelihood_h_1,prior_h_1,b.posterior_h_1,prior_h_2,posterior_h_2")
        for o in DEBUG_RAW:
            f.write(o + "\n")

    print("Results written successfully")

for i in range(100):
    main()
