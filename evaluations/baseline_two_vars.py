#!/usr/bin/python3

import os
import shutil
from decimal import *
import random

"""
Industrial control systems, computer software and certainly sophisticated malware are subject 
to many different input values, and so we attempt to simulate two competing pieces of evidence 
for the same hypothesis H_1, one of high probability and one of low.
"""


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


def iterative(iterations, new_h_1, p):
    """
    Continue to update in the same one fashion until convergence
    So for each iteration above our belief change, we recalculate likelihoods
    """

    likelihood_h_1 = .51
    prior_h_1 = .50
    likelihood_h_2 = .49
    prior_h_2 = .50

    items = []
    print("")
    for i in range(iterations):
        print("Processing iterative" + str(i), end='\r')

        has_processed = False
        if(percent_of_the_time(p)):
            likelihood_h_1 = new_h_1
            likelihood_h_2 = 1 - likelihood_h_1
            has_processed = True

        if(not has_processed):
            likelihood_h_1 = .51
            likelihood_h_2 = .49

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


print("""
 ________  ____    ____
|_   __  ||_   \  /   _|
  | |_ \_|  |   \/   |
  |  _| _   | |\  /| |
 _| |__/ | _| |_\/_| |_
|________||_____||_____|
                         """)

getcontext().prec = 5

ITERATIONS = 200
PRIOR_H_1 = .5
PRIOR_H_2 = .5

DEBUG = True
DEBUG_RAW = []
DEBUG_REEVALS = []

# i want to build the array of E and their likelihoods here, and pass them into a simple function of calculation....

# results1 = bayes(.51, PRIOR_H_1, .49, PRIOR_H_2, ITERATIONS)
# results2 = iterative(ITERATIONS, .49, 10)
# results3 = iterative(ITERATIONS, .49, 20)
# results4 = iterative(ITERATIONS, .49, 30)
# results5 = iterative(ITERATIONS, .49, 40)
# results6 = iterative(ITERATIONS, .49, 50)
# results7 = iterative(ITERATIONS, .49, 60)
# results8 = iterative(ITERATIONS, .49, 70)
# results9 = iterative(ITERATIONS, .49, 80)
# results10 = iterative(ITERATIONS, .49, 90)
# results11 = iterative(ITERATIONS, .49, 100)


results1 = bayes(.51, PRIOR_H_1, .49, PRIOR_H_2, ITERATIONS)
results2 = iterative(ITERATIONS, .49, 10)
results3 = iterative(ITERATIONS, .4, 10)
results4 = iterative(ITERATIONS, .35, 10)
results5 = iterative(ITERATIONS, .3, 10)
results6 = iterative(ITERATIONS, .25, 10)
results7 = iterative(ITERATIONS, .2, 10)
results8 = iterative(ITERATIONS, .15, 10)
results9 = iterative(ITERATIONS, .1, 10)
results10 = iterative(ITERATIONS, .05, 10)
results11 = iterative(ITERATIONS, .01, 10)


if os.path.isdir("output"):
    shutil.rmtree("output")
os.makedirs("output")

with open("output/results.txt", "w") as f:
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

with open("output/output_reevals.txt", "w") as f:
    f.write("causing_evidence_postion, prior_evidence_positions_to_be_updated, variance_h_1, variance_h_2")
    for r in DEBUG_REEVALS:
        f.write(str(r.causing_evidence_postion) + "," + str(r.prior_evidence_positions_to_be_updated) + "," + str(r.variance_h_1) + "," + str(r.variance_h_2) + "\n")

with open("output/output_raw_debug.txt", "w") as f:
    f.write("i,iter,likelihood_h_1,prior_h_1,b.posterior_h_1,prior_h_2,posterior_h_2")
    for o in DEBUG_RAW:
        f.write(o + "\n")

print("Results written successfully")
exit(0)
