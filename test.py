from classes import SalesModel
from classes import PigGrowthModel
from classes import BarnModel

import numpy
import matplotlib.pyplot as plt
from scipy.integrate import quad

w2fModel = [-203.950303, 288.798017, -81.388061, 10.101958, -0.623565, 0.018835, -0.000222]
finModel = [-130.739421, 210.431209, -65.244076, 9.294793, -0.678313, 0.024900, -0.000365]
awgModel = [-0.87031545, 2.16250341, -0.09743488, 0.00122924]
awfcModel = [1, 0.10728206]
awgAdjust = [273, 0, 26]
fccumAdjust = [2.57, 0, 26]

wtCutoff = [12, 15, 21,
            50, 80, 120, 160,
            200, 225, 245, 265]

priceCutoff = [0.29721160, 0.19111814, 0.11239397,
               0.09492490, 0.08964955, 0.08389767, 0.08180934,
               0.08025336, 0.07925855, 0.09876173, 0.10228566]

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

gm = PigGrowthModel(awgModel, awfcModel, w2fModel, 12, awgAdjust, fccumAdjust, priceCutoff, wtCutoff)
bm = BarnModel(w2fModel, gm)
print bm.feed_cost_total.integrate(0,26)


##x = numpy.arange(0,27,.1)
##plt.plot(x, bm.death.model(x))
##plt.plot(x, bm.death_shift.model(x))
##plt.plot(x, bm.alive.model(x))
##plt.plot(x, bm.alive_shift.model(x))
##plt.show()
