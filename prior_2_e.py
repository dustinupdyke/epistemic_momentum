#!/usr/bin/python3

import os
import shutil
from decimal import *


class Evidence:
    """
    Basic unit of this model, iterations of evidence E
    """

    def __init__(self, position, likelihood_h_1, likelihood_h_2, prior_h_1, prior_h_2, prior_evidence_positions_to_be_updated, reevaluation_likelihood_h_1, reevaluation_likelihood_h_2):
        self.position = position
        self.likelihood_h_1 = likelihood_h_1
        self.likelihood_h_2 = likelihood_h_2
        self.prior_h_1 = prior_h_1
        self.prior_h_2 = prior_h_2
        self.prior_evidence_positions_to_be_updated = prior_evidence_positions_to_be_updated
        self.reevaluation_likelihood_h_1 = reevaluation_likelihood_h_1
        self.reevaluation_likelihood_h_2 = reevaluation_likelihood_h_2


class BayesItem:
    """
    The object for bayes calculations
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


def lookback(evidences, is_debug=False):
    """
    For the reevaluation of previous evidence
    """
    items = []
    for i, evidence in enumerate(evidences):

        is_debug = False
        if i == 50 or i == 75:
            is_debug = True

        if i == 0:
            prior_h_1 = evidence.prior_h_1
            prior_h_2 = evidence.prior_h_2

        original_likelihood_h_1 = evidence.likelihood_h_1
        original_likelihood_h_2 = evidence.likelihood_h_2

        if is_debug:
            DEBUG_RAW.append(evidence.__dict__)

        if evidence.prior_evidence_positions_to_be_updated:

            # go back and reevaluate all items to this point
            for update_count, item in enumerate(items):

                # previous E originals
                likelihood_h_1 = item.likelihood_h_1
                likelihood_h_2 = item.likelihood_h_2

                if update_count == 0:
                    prior_h_1 = item.prior_h_1
                    prior_h_2 = item.prior_h_2

                # is this a previous E to reevaluate?
                if update_count in evidence.prior_evidence_positions_to_be_updated:
                    likelihood_h_1 = normalize(evidence.reevaluation_likelihood_h_1)
                    likelihood_h_2 = normalize(evidence.reevaluation_likelihood_h_2)

                lookback_item = BayesItem(update_count, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)

                if is_debug:
                    DEBUG_RAW.append("{0:03d}".format(i) + "," +
                                     "{0:03d}".format(update_count) + "," +
                                     str(likelihood_h_1) + "," +
                                     "{:5f}".format(prior_h_1) + "," +
                                     "{:5f}".format(lookback_item.posterior_h_1) + "," +
                                     "{:5f}".format(prior_h_2) + "," +
                                     "{:5f}".format(lookback_item.posterior_h_2))

                prior_h_1 = lookback_item.posterior_h_1
                prior_h_2 = lookback_item.posterior_h_2

        likelihood_h_1 = original_likelihood_h_1
        likelihood_h_2 = original_likelihood_h_2

        b = BayesItem(i, likelihood_h_1, prior_h_1, likelihood_h_2, prior_h_2)
        items.append(b)

        if is_debug:
            DEBUG_RAW.append("{0:03d}".format(i) + "," +
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
________________________________________________________________________________________
 ________  ____    ____
|_   __  ||_   \  /   _|
  | |_ \_|  |   \/   |
  |  _| _   | |\  /| |
 _| |__/ | _| |_\/_| |_
|________||_____||_____|
________________________________________________________________________________________                         

""")

getcontext().prec = 5

ITERATIONS = 200
PRIOR_H_1 = .5
PRIOR_H_2 = .5

DEBUG = True
DEBUG_RAW = []
DEBUG_EVIDENCE = []

evidences1 = []
evidences2 = []
evidences3 = []
evidences4 = []
evidences5 = []
reevaluations = []
evidence_to_reeval1 = []
evidence_to_reeval2 = []
evidence_to_reeval3 = []
evidence_to_reeval4 = []
evidence_to_reeval5 = []
for i in range(ITERATIONS):

    evidences1.append(Evidence(i, .51, .49, .5, .5, [], .51, .49))

    if(i % 5 == 0):
        evidence_to_reeval2.append(i)
    if(i % 13 == 0):
        evidence_to_reeval3.append(i)
    if(i % 26 == 0):
        evidence_to_reeval4.append(i)
    if(i % 51 == 0):
        evidence_to_reeval5.append(i)

    has_processed = False
    if(i > 149):
        evidences2.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval2[-3:-1], .51, .49))
        evidences3.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval3[-3:-1], .51, .49))
        evidences4.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval4[-3:-1], .51, .49))
        evidences5.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval5[-3:-1], .51, .49))
        has_processed = True
    if(i > 99 and not has_processed):
        evidences2.append(Evidence(i, .48, .52, .5, .5, evidence_to_reeval2[-3:-1], .48, .52))
        evidences3.append(Evidence(i, .48, .52, .5, .5, evidence_to_reeval3[-3:-1], .48, .52))
        evidences4.append(Evidence(i, .48, .52, .5, .5, evidence_to_reeval4[-3:-1], .48, .52))
        evidences5.append(Evidence(i, .48, .52, .5, .5, evidence_to_reeval5[-3:-1], .48, .52))
        has_processed = True
    if(i > 49 and not has_processed):
        evidences2.append(Evidence(i, .49, .51, .5, .5, evidence_to_reeval2[-3:-1], .49, .51))
        evidences3.append(Evidence(i, .49, .51, .5, .5, evidence_to_reeval3[-3:-1], .49, .51))
        evidences4.append(Evidence(i, .49, .51, .5, .5, evidence_to_reeval4[-3:-1], .49, .51))
        evidences5.append(Evidence(i, .49, .51, .5, .5, evidence_to_reeval5[-3:-1], .49, .51))
        has_processed = True
    if(not has_processed):
        evidences2.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval2[-3:-1], .51, .49))
        evidences3.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval3[-3:-1], .51, .49))
        evidences4.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval4[-3:-1], .51, .49))
        evidences5.append(Evidence(i, .51, .49, .5, .5, evidence_to_reeval5[-3:-1], .51, .49))

if DEBUG:
    for e in evidences1:
        DEBUG_EVIDENCE.append(e.__dict__)
    for e in evidences2:
        DEBUG_EVIDENCE.append(e.__dict__)
    for e in evidences3:
        DEBUG_EVIDENCE.append(e.__dict__)
    for e in evidences4:
        DEBUG_EVIDENCE.append(e.__dict__)
    for e in evidences5:
        DEBUG_EVIDENCE.append(e.__dict__)

results0 = bayes(.51, PRIOR_H_1, .49, PRIOR_H_2, ITERATIONS)

results1 = lookback(evidences1, DEBUG)
results2 = lookback(evidences2, DEBUG)
results3 = lookback(evidences3, DEBUG)
results4 = lookback(evidences4, DEBUG)
results5 = lookback(evidences5, DEBUG)

shutil.rmtree("output")
os.makedirs("output")

with open("output/results.txt", "w") as f:
    # f.write("E  ,Bayes,Single,Iterative,LookbackSingle,LookbackIterative\n")
    for i in range(ITERATIONS):
        f.write("{0:03d}".format(i) + "," +
                "{:5f}".format(results0[i].posterior_h_1) + "," +
                "{:5f}".format(results1[i].posterior_h_1) + "," +
                "{:5f}".format(results2[i].posterior_h_1) + "," +
                "{:5f}".format(results3[i].posterior_h_1) + "," +
                "{:5f}".format(results4[i].posterior_h_1) + "," +
                "{:5f}".format(results5[i].posterior_h_1) + "\n")

with open("output/output_evidence.txt", "w") as f:
    for o in DEBUG_EVIDENCE:
        f.write(str(o) + "\n")

with open("output/output_raw_debug.txt", "w") as f:
    for o in DEBUG_RAW:
        f.write(str(o) + "\n")

print("Results written successfully")
