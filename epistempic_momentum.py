#!/usr/bin/python3

import matplotlib.pyplot as plt
from decimal import *
import random
from copy import deepcopy


class BayesItem:
    def __init__(self, position, likelihood_H_1, prior_H_1, likelihood_H_2, prior_H_2):
        self.Position = position
        self.Likelihood_H_1 = likelihood_H_1
        self.Likelihood_H_2 = likelihood_H_2
        self.Prior_H_1 = prior_H_1
        self.Prior_H_2 = prior_H_2
        self.Posterior_H_1 = 0.0
        self.Posterior_H_2 = 0.0
        self.Calculate_Posterior()

    def Calculate_Posterior(self):
        #                    Likelihood(H_1)	    Prior(H_1)	      Likelihood(MG)	      Prior(H1)	        Likelihood(MB)	       Prior(H2)
        self.Posterior_H_1 = (self.Likelihood_H_1 * self.Prior_H_1) / ((self.Likelihood_H_1 * self.Prior_H_1) + (self.Likelihood_H_2 * self.Prior_H_2))
        self.Posterior_H_2 = (self.Likelihood_H_2 * self.Prior_H_2) / ((self.Likelihood_H_2 * self.Prior_H_2) + (self.Likelihood_H_1 * self.Prior_H_1))

        self.Posterior_H_1 = normalize(self.Posterior_H_1)
        self.Posterior_H_2 = normalize(self.Posterior_H_2)


def IterativeBayes(Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2, iterations):
    items = []
    for i in range(iterations):
        b = BayesItem(i, Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2)
        items.append(b)
        Prior_H_1 = b.Posterior_H_1
        Prior_H_2 = b.Posterior_H_2
    return items


def IterativeSingleUpdate(Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2, iterations, iterationToUpdate, varianceLow, varianceHigh):
    items = []
    for i in range(iterations):
        b = BayesItem(i, Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2)
        items.append(b)
        if i == iterationToUpdate:
            p = random.uniform(varianceLow, varianceHigh)
            Prior_H_1 = b.Posterior_H_1 + p
            Prior_H_2 = b.Posterior_H_2 + (1-abs(p))
        else:
            Prior_H_1 = b.Posterior_H_1
            Prior_H_2 = b.Posterior_H_2
    return items


def IterativeReEvaluation(Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2, iterations, iterationToUpdate, varianceLow, varianceHigh):
    items = []
    for i in range(iterations):
        b = BayesItem(i, Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2)
        items.append(b)

        if i > iterationToUpdate:
            p = random.uniform(varianceLow, varianceHigh)
            updated_item_prior_h1 = b.Posterior_H_1 + p
            updated_item_prior_h2 = b.Posterior_H_2 + (1-abs(p))

            updated_item_prior_h1 = normalize(updated_item_prior_h1)
            updated_item_prior_h2 = normalize(updated_item_prior_h2)

            for update_count, item in enumerate(items):
                updated_item = deepcopy(item)

                updated_item.Prior_H_1 = updated_item_prior_h1
                updated_item.Prior_H_2 = updated_item_prior_h2

                updated_item.Calculate_Posterior()

                b.Prior_H_1 = updated_item.Posterior_H_1
                b.Prior_H_2 = updated_item.Posterior_H_2

                print(str(i) + "|" + str(update_count) + " " + str(b.Prior_H_1))

            b.Calculate_Posterior()

            print(str(i) + " " + str(b.Prior_H_1) + " " + str(b.Posterior_H_1))

            exit(1)

            """
            1
            11
            111
            1111
            11111
            111111
            1111111
            22222222
            333333333

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
            """

            # print("{:5f}".format(b.Posterior_H_1) +
            #       " " + "{:f}".format(p) + " " "{:5f}".format(b.Posterior_H_1 + p))
            Prior_H_1 = b.Posterior_H_1
            Prior_H_2 = b.Posterior_H_2
        else:
            Prior_H_1 = b.Posterior_H_1
            Prior_H_2 = b.Posterior_H_2

    return items


def normalize(n):
    if n > 1:
        n = 1
    if n < 0:
        n = 0
    return n


getcontext().prec = 5

iterations = 65
iterationToStartReEvaluation = 6
reEvaluationLow = -.3
reEvaluationHigh = -.3
likelihood_H_1 = .7
likelihood_H_2 = .3
prior_H_1 = .5
prior_H_2 = .5


results1 = IterativeBayes(likelihood_H_1, prior_H_1, likelihood_H_2, prior_H_2,
                          iterations)
# for i, result in enumerate(results1):
# print(str(i) + "," + "{:5f}".format(result.Posterior_H_1) + "," + "{:5f}".format(result.Prior_H_1))
# print("{:5f}".format(result.Posterior_H_1))

results2 = IterativeSingleUpdate(likelihood_H_1, prior_H_1, likelihood_H_2, prior_H_2,
                                 iterations, iterationToStartReEvaluation, reEvaluationLow, reEvaluationHigh)
# for i, result in enumerate(results2):
# print(str(i) + "," + "{:5f}".format(result.Posterior_H_1) + "," + "{:5f}".format(result.Prior_H_1))
# print("{:5f}".format(result.Posterior_H_1)

results3 = IterativeReEvaluation(likelihood_H_1, prior_H_1, likelihood_H_2, prior_H_2,
                                 iterations, iterationToStartReEvaluation, reEvaluationLow, reEvaluationHigh)
# for i, result in enumerate(results3):
# print(str(i) + "," + "{:5f}".format(result.Posterior_H_1) +
#       "," + "{:5f}".format(result.Prior_H_1))
# print("{:5f}".format(result.Posterior_H_1))

for i in range(iterations):
    print(str(i) + "," + "{:5f}".format(results1[i].Posterior_H_1) + "," + "{:5f}".format(
        results2[i].Posterior_H_1) + "," + "{:5f}".format(results3[i].Posterior_H_1))

exit(1)


# ------------------

x = []
y = []
x2 = []
y2 = []
i = 0
for result in results:
    print("{:5f}".format(result.Posterior_H_1) +
          "," + "{:5f}".format(result.Posterior_H_2))

    # x axis values
    y.append("{:2f}".format(result.Posterior_H_1))
    # corresponding y axis values
    x.append(i)
    y2.append("{:2f}".format(result.Posterior_H_2))
    # corresponding y axis values
    x2.append(i)
    i += 1

# plotting the points

plt.scatter(x, y)
plt.scatter(x2, y2)
plt.plot(x, y, label="Posterior_H_1")
plt.plot(x2, y2, label="Posterior_H_2")
plt.legend()


plt.ylim(0, 25)
plt.xlim(0, 25)


# naming the x axis
plt.xlabel("Evidence Position")
# naming the y axis
plt.ylabel("Belief")

# giving a title to my graph
plt.title("EM")

# function to show the plot
plt.show()
