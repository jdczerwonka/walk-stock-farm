CARGILL_WB = "C:\Users\Jonathan.WSF\Documents\GitHub\walk-stock-farm\Cargill Lean Value Matrix.csv"

from classes import SalesModel
from classes import PigGrowthModel
from classes import BarnModel

import math
import numpy
import pandas
from scipy.stats import norm
from scipy.stats import logistic

from scipy.integrate import quad
from scipy.optimize import newton

import matplotlib.pyplot as plt

polynomial = numpy.polynomial.polynomial.Polynomial

w2fModel = [0, 288.798017, -81.388061, 10.101958, -0.623565, 0.018835, -0.000222]
finModel = [0, 210.431209, -65.244076, 9.294793, -0.678313, 0.024900, -0.000365]
awgModel = [0.43503557, 2.16250341, -0.09743488, 0.00122924]
awfiModel = [0.90582088, 1.59691733, 0.24820408, -0.01655183, 0.00028117]
awfcModel = [1.1, 0.10728206]
awgAdjust = [273, 0, 24.5]
awfcAdjust = [2.63, 0, 24.5]

wtCutoff = numpy.array([12, 20, 30,
            50, 80, 120, 160,
            200, 225, 245, 265])

priceCutoff = [0.29721160, 0.19111814, 0.11239397,
               0.09492490, 0.08964955, 0.08389767, 0.08180934,
               0.08025336, 0.07925855, 0.09876173, 0.10228566]

priceCutoff2 = [0.31444049, 0.21165624, 0.13395160,
				0.09024673, 0.09004769, 0.08920827, 0.08843443,
				0.08818230, 0.08798901, 0.09244352, 0.09587957]

# priceCutoffLB = numpy.array([0, 0.31444049, 0.21165624, 0.13395160,
# 				0.09024673, 0.09004769, 0.08920827, 0.08843443,
# 				0.08818230, 0.08798901, 0.09244352])

##x = []
##y = []
##for price in range(55,105, 5):
##    sm.base_price = price
##    x.append(price)
##    df = sm.calculate_range(range(250, 325, 5), "live_avg")
##    print price
##    print df
##    y.append(df["live_avg"][df["rev_net_pig"].idxmax()])
##
##plt.plot(x,y)
##plt.show()

# packer_matrix_df = pandas.read_csv(CARGILL_WB, header=None)
# print packer_matrix_df[1]

sm = SalesModel(CarcassAvg = 225, CarcassStdDev = 17.6, LeanAvg = 54.61,
               LeanStdDev = 2.11, YieldAvg = 76.35, BasePrice = 51)
gm = PigGrowthModel(awgModel, awfcModel, awgAdjust, awfcAdjust, priceCutoff2, wtCutoff)
bm = BarnModel(w2fModel, gm, sm, StartWeight = 12, BarnSize = 170, ModelType = "cut")


# x = numpy.arange(0, 26.1, .1)
# plt.plot(x, bm.feed_cost_cum.model(x))
# plt.show()


# x = numpy.arange(250, 295, 5)
# plt.plot(x, bm.calc_rev_curve(x))
# plt.show()



x = numpy.arange(50, 111, 1)
plt.plot(x, bm.calc_opt_price_curve(x))
plt.show()

# print bm.aw_feed_cost.integrate(0, 26)

# print gm.fc_cum.model(23.1)
# print gm.g_cum.model(23.1)
# print gm.fi_cum.model(23.1)

# print gm.aw_feed_cost.integrate(0, 23.1)
