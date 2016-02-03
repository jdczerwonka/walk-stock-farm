from classes import SalesModel
from classes import PigGrowthModel
from classes import BarnModel

import numpy
import matplotlib.pyplot as plt
from scipy.integrate import quad

polynomial = numpy.polynomial.polynomial.Polynomial 

w2fModel = [0, 288.798017, -81.388061, 10.101958, -0.623565, 0.018835, -0.000222]
finModel = [0, 210.431209, -65.244076, 9.294793, -0.678313, 0.024900, -0.000365]
awgModel = [0.43503557, 2.16250341, -0.09743488, 0.00122924]
awfiModel = [0.90582088, 1.59691733, 0.24820408, -0.01655183, 0.00028117]
awfcModel = [1.1, 0.10728206]
awgAdjust = [273, 0, 23.1]
awfcAdjust = [2.57, 0, 23.1]

wtCutoff = [12, 20, 30,
            50, 80, 120, 160,
            200, 225, 245, 265]

priceCutoff = [0.29721160, 0.19111814, 0.11239397,
               0.09492490, 0.08964955, 0.08389767, 0.08180934,
               0.08025336, 0.07925855, 0.09876173, 0.10228566]

priceCutoff2 = [0.31444049, 0.21165624, 0.13395160,
				0.09024673, 0.09004769, 0.08920827, 0.08843443,
				0.08818230, 0.08798901, 0.09244352, 0.09587957]

##sm = SalesModel(MarketSize = 170, CarcassAvg = 229, CarcassStdDev = 14, LeanAvg = 54.2,
##                LeanStdDev = 2.17, YieldAvg = 76.2, BasePrice = 80)
##
##sm.growth_model = GrowthModel(awgModel, awfcModel, w2fModel, 12, awgAdjust, fccumAdjust)
##sm.adjust_death_model
####sm.calculate_model()
####print sm.carcass_df
##
##df = sm.calculate_range(range(250, 325, 5), "live_avg")
##plt.plot(df["live_avg"], df["rev_net_pig"])
##plt.show()    

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

gm = PigGrowthModel(awgModel, awfcModel, awgAdjust, awfcAdjust, priceCutoff, wtCutoff)
bm = BarnModel(w2fModel, gm, BarnSize = 170)

print gm.aw_feed_cost.integrate(0, 23.1)
print bm.aw_feed_cost.integrate(0, 23.1)
print bm.calc_feed_cost_diff(270, 290, "wt")

# x = numpy.arange(0, 24.6, .1)
# plt.plot(x, bm.aw_feed_cost.model(x))
# plt.show()

# print gm.fc_cum.model(23.1)
# print gm.fi_cum.model(23.1) / gm.g_cum.model(23.1)
# print gm.awfc.integrate(0,23.1) / 23.1

# print gm.fc_cum.model(23.1)
# print gm.g_cum.model(23.1)
# print gm.fi_cum.model(23.1)

# print gm.aw_feed_cost.integrate(0, 23.1)
