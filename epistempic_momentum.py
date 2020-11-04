#!/usr/bin/python3

import matplotlib.pyplot as plt
from decimal import *


class Bayes_Item:
    def __init__(self, likelihood_H_1, prior_H_1, likelihood_H_2, prior_H_2):
        self.Likelihood_H_1 = likelihood_H_1
        self.Likelihood_H_2 = likelihood_H_2
        self.Prior_H_1 = prior_H_1
        self.Prior_H_2 = prior_H_2
        self.Posterior_H_1 = 0.0
        self.Posterior_H_2 = 0.0
        self.Calculate_Posterior()

    def Calculate_Posterior(self):
        #   Likelihood(H_1)	Prior(H_1)	Likelihood(MG)	Prior(H1)	Likelihood(MB)	Prior(H2)
        # =(D15*E15)/(F15*G15+(H15*I15))
        self.Posterior_H_1 = (self.Likelihood_H_1 * self.Prior_H_1) / \
            ((self.Likelihood_H_1 * self.Prior_H_1) +
             (self.Likelihood_H_2 * self.Prior_H_2))

        self.Posterior_H_2 = (self.Likelihood_H_2 * self.Prior_H_2) / \
            ((self.Likelihood_H_2 * self.Prior_H_2) +
             (self.Likelihood_H_1 * self.Prior_H_1))


def IterativeBayes(Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2, iterations):
    items = []

    for _ in range(iterations):
        b = Bayes_Item(Likelihood_H_1, Prior_H_1, Likelihood_H_2, Prior_H_2)

        items.append(b)
        Prior_H_1 = b.Posterior_H_1
        Prior_H_2 = b.Posterior_H_2
    return items


getcontext().prec = 5

results = IterativeBayes(.9, .5, .1, .5, 20)

x = []
y = []
x2 = []
y2 = []
i = 0
for result in results:
    print('{:5f}'.format(result.Posterior_H_1) +
          "," + '{:5f}'.format(result.Posterior_H_2))

    # x axis values
    y.append('{:2f}'.format(result.Posterior_H_1))
    # corresponding y axis values
    x.append(i)
    y2.append('{:2f}'.format(result.Posterior_H_2))
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
