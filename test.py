CARGILL_WB = "C:\Users\Jonathan.WSF\Documents\GitHub\walk-stock-farm\Cargill Lean Value Matrix.csv"

from classes import SalesModel
from classes import PigGrowthModelPiece
from classes import BarnModelPiece
from classes import PiecePolynomial

import math
import numpy
import pandas
from scipy.stats import norm
from scipy.stats import logistic

from scipy.integrate import quad
from scipy.optimize import newton
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
import simplejson

polynomial = numpy.polynomial.polynomial.Polynomial

w2fDeath1 = [-0.2500, 23.9464, -4.30357143]
w2fDeath2 = [11]
w2fDeath3 = [-526.6290 + 11, 66.8357, -2.03571429]
w2fDeath4 = [-725.9390 + 11, 64.3674, -1.35984848]
w2fDeath5 = [0]

w2fDeath_br = [0, 5, 13.1, 19.1, 27]

finModel = [0, 210.431209, -65.244076, 9.294793, -0.678313, 0.024900, -0.000365]
awgModel = [0.43503557, 2.16250341, -0.09743488, 0.00122924]
awfiModel = [0.90582088, 1.59691733, 0.24820408, -0.01655183, 0.00028117]
awfcModel = [1.1, 0.10728206]
awgAdjust = [273, 0, 24.5]
awfcAdjust = [2.65, 24.5]

# wtCutoff = numpy.array([12, 20, 30,
#             50, 80, 120, 160,
#             200, 225, 245, 265])

feed_cost_br = [2.62141065, 4.12473822, 
				6.29822741, 8.85710311, 11.78041278, 14.48593525,
				17.1341315, 17.63894897, 18.8702043, 20.1224158]

feed_cost_br_wt = [ 20., 30.,
					50., 80., 120., 160.,
            		200., 225., 245., 265.]

priceCutoff = [	0.29438071, 0.18980649, 0.11205248,
				0.09077630, 0.08552250, 0.07965624, 0.07812813,
				0.07665118, 0.07570529, 0.09360718, 0.09719425]


awg_poly = [polynomial(awgModel), polynomial(awgModel) * (1 + (0.089 * 1)), polynomial(awgModel) * (1 + (0.089 * 0.8))]
awg_br = [21., 23.]
awg_br_wt = [245., 265.]
awgModel = PiecePolynomial(awg_poly, awg_br, awg_br_wt)

awfc_poly = [polynomial(awfcModel), polynomial(awfcModel) * (1 - (0.142 * 1)), polynomial(awfcModel) * (1 - (0.142 * 0.8))]
awfc_br = [21., 23.]
awfcModel = PiecePolynomial(awfc_poly, awfc_br)

feed_cost_poly = []
for cost in priceCutoff:
	feed_cost_poly.append(polynomial([cost]))

feedCostModel = PiecePolynomial(feed_cost_poly, feed_cost_br, feed_cost_br_wt)

death_poly = [polynomial(0), polynomial(w2fDeath1), polynomial(w2fDeath2), polynomial(w2fDeath3), polynomial(w2fDeath4), polynomial(w2fDeath5)]
death_br = w2fDeath_br

deathModel = PiecePolynomial(death_poly, death_br)

rentModel = PiecePolynomial([polynomial(0), polynomial(2030)], [0.])

sm = SalesModel(CarcassAvg = 218, CarcassStdDev = 19, LeanAvg = 54.30, LeanStdDev = 2.11, YieldAvg = 76.29, BasePrice = 71.64)
gm = PigGrowthModelPiece(awgModel, awfcModel, feedCostModel, awgAdjust, awfcAdjust, 12)
bm = BarnModelPiece(deathModel, rentModel, gm, sm, StartNum = 2500, DeathLossPer = 3.25, DiscountLossPer = 0.75, DiscountPricePer = 50, AvgWeeksInBarn = 24.5)

# x = numpy.arange(0, 24.6, .1)
# plt.plot(x, gm.feed_cost_total(x))
# # plt.plot(x, bm.feed_cost_total(x))
# # plt.plot(x, bm.death(x))
# # plt.plot(x, bm.death_total(x))
# # plt.plot(x, bm.alive_total(x))
# plt.show()


x = numpy.arange(265, 290.1, .1)
plt.plot(x, bm.calc_rev_curve(x))
plt.show()