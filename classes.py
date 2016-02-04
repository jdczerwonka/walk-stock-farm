CARGILL_WB = "C:\Users\Jonathan.WSF\Desktop\Market Sales Model\Cargill Lean Value Matrix.csv"
WT_CUTOFF_DEFAULT = [50, 80, 120, 160, 200, 225, 245, 265]

import math
import numpy
import pandas
from scipy.stats import norm
from scipy.stats import logistic

from scipy.integrate import quad
from scipy.optimize import newton

polynomial = numpy.polynomial.polynomial.Polynomial

class SalesModel:
    def __init__(self, CarcassAvg = 220, CarcassStdDev = 18,
                 LeanAvg = 54, LeanStdDev = 2.1, YieldAvg = 76, BasePrice = 80,
                 Packer = "Cargill", LeanDist = "norm", CarcassDist = "logistic"):
        
        self.carcass_avg = CarcassAvg
        self.carcass_std_dev = CarcassStdDev

        self.lean_avg = LeanAvg
        self.lean_std_dev = LeanStdDev
        self.yield_avg = YieldAvg

        self.live_avg = self.carcass_avg / (self.yield_avg / 100)
        
        self.base_price = BasePrice

        self.packer = Packer
        self.lean_dist = LeanDist
        self.carcass_dist = CarcassDist

        self.calc_matrix_factor
        self.calc_revenue

    @property
    def carcass_s(self):
        return math.sqrt(3 * self.carcass_std_dev ** 2 / math.pi ** 2)

    @property 
    def calc_matrix_factor(self):
        if self.lean_dist == "norm":
            prob_dist = norm
            std_dev = self.lean_std_dev

        if self.packer == "Cargill":
            self.packer_wt_arr = [0.5, 140.5, 147.5, 154.5, 162.5, 169.5, 176.5, 184.5, 191.5, 198.5, 206.5, 213.5, 221.5, 228.5, 235.5, 242.5, 244.5, 249.5, 256.5, 263.5]
            self.packer_matrix_df = pandas.read_csv(CARGILL_WB, header=None)

            self.matrix_factor = []
            for j in range(0, 20):
                tot = 0
                for i in range(0, 24):
                    tot = tot + self.packer_matrix_df.iloc[i, j] * (prob_dist.cdf(39.5 + 1 + i, self.lean_avg, std_dev) - prob_dist.cdf(39.5 + i, self.lean_avg, std_dev))

                self.matrix_factor.append( tot + self.packer_matrix_df.iloc[0, j] * (prob_dist.cdf(39.5, self.lean_avg, std_dev) - prob_dist.cdf(-numpy.inf, self.lean_avg, std_dev)) +
                    self.packer_matrix_df.iloc[0, j] * (prob_dist.cdf(numpy.inf, self.lean_avg, std_dev) - prob_dist.cdf(63.5, self.lean_avg, std_dev)) )

    @property 
    def calc_revenue(self):        
        if self.carcass_dist == "norm":
            prob_dist = norm
            std_dev = self.carcass_std_dev
        elif self.carcass_dist == "logistic":
            prob_dist = logistic
            std_dev = self.carcass_s

        self.base_price_adj = 0
        for i in range(0, len(self.matrix_factor) - 1):
            self.base_price_adj = self.base_price_adj + ( self.matrix_factor[i] + self.base_price ) * ( (prob_dist.cdf(self.packer_wt_arr[i + 1], self.carcass_avg, std_dev) - prob_dist.cdf(self.packer_wt_arr[i], self.carcass_avg, std_dev)) / 100 )

        self.base_price_adj = ( self.base_price_adj + ( self.matrix_factor[0] + self.base_price ) * ( (prob_dist.cdf(self.packer_wt_arr[0], self.carcass_avg, std_dev) - prob_dist.cdf(-numpy.inf, self.carcass_avg, std_dev)) / 100 ) +
            ( self.matrix_factor[len(self.matrix_factor) - 1] + self.base_price ) * ( (prob_dist.cdf(numpy.inf, self.carcass_avg, std_dev) - prob_dist.cdf(self.packer_wt_arr[len(self.matrix_factor) - 1], self.carcass_avg, std_dev)) / 100 ) )

        self.revenue_avg = self.base_price_adj * self.carcass_avg

class PigGrowthModel():
    def __init__(self, awgModel = [0], awfcModel = [0],
                 awgAdjust = None, awfcAdjust = None, 
                 PriceCutoff = [0], WtCutoff = WT_CUTOFF_DEFAULT, WeekCutoff = None,
                 StartWeight = 12):

        self.start_weight = StartWeight

        self.awg = Model( polynomial(awgModel) )
        if awgAdjust is not None:
            self.shift_awg(awgAdjust[0], awgAdjust[1], awgAdjust[2])

        self.set_g_total
        self.set_g_cum

        self.awfc = Model( polynomial(awfcModel) )
        if awfcAdjust is not None:
            self.shift_awfc(awfcAdjust[0], awfcAdjust[1], awfcAdjust[2])

        self.set_fc_cum
        self.set_awfi
        self.set_fi_total
        self.set_fi_cum

        self.cutoff_price = PriceCutoff
        self.cutoff_wt = WtCutoff

        if WeekCutoff is None:
            self.calc_week_cutoff

        self.feed_cost = Model(lambda x: self.heaviside_combined(x))
        self.aw_feed_cost = Model(lambda x: self.feed_cost.model(x) * self.awfi.model(x))

    def shift_awg(self, zero, lb, ub):
        self.awg = Model( self.awg.model * newton(lambda x: quad(self.awg.model * x, lb, ub)[0] - zero, 1))

    def shift_awfc(self, zero, lb, ub):
        # self.awfc = Model( polynomial( [self.awfc.model.coef[0], newton(lambda x: self.awfc.integrate(lb, ub, ub - lb, x) - zero, 0)] ) )
        self.awfc = Model( polynomial( [self.awfc.model.coef[0], newton(lambda x: self.opt_fc(x, ub) - zero, 0)] ) )

    def opt_fc(self, x, wk):
        self.awfc.model.coef[1] = x
        self.set_awfi
        self.set_fi_cum
        self.set_fc_cum
        return self.fc_cum.model(wk)

    @property
    def set_g_total(self):
        p = self.awg.model.integ()
        self.g_total = Model( polynomial(p.coef) )

    @property 
    def set_g_cum(self):
        p = self.awg.model.integ()
        m = polynomial(p.coef)
        self.g_cum = Model(lambda x: m(x) / (x * 7) )

    @property
    def set_fc_cum(self):
        self.fc_cum = Model(lambda x: self.fi_cum.model(x) / self.g_cum.model(x))

    @property
    def set_awfi(self):
        p = self.awg.model * self.awfc.model
        self.awfi = Model( polynomial(p.coef) )

    @property
    def set_fi_total(self):
        p = self.awfi.model.integ()
        self.awfi_total = Model( polynomial(self.awfi) ) 

    @property 
    def set_fi_cum(self):
        p = self.awfi.model.integ()
        m = polynomial(p.coef)
        self.fi_cum = Model(lambda x: m(x) / (x * 7) )

    @property
    def calc_week_cutoff(self):
        self.cutoff_week = []
        for wt in self.cutoff_wt:
            self.cutoff_week.append(self.awg.calc_week(wt - self.start_weight))

        self.cutoff_week.append(26)

    def heaviside(self, x):
        return 0.5 + 0.5 * numpy.tanh(5000 * x)

    def heaviside_combined(self, x):
        func = self.cutoff_price[0] * self.heaviside(x - self.cutoff_week[0])
        for i in range(1, len( self.cutoff_week ) - 1, 1):
            func = func + (self.cutoff_price[i] - self.cutoff_price[i - 1]) * self.heaviside(x - self.cutoff_week[i])  

        return func 

class BarnModel():
    def __init__(self, DeathModel, PigModel, SalesModel, StartWeight = 12, BarnSize = 2500, DeathLossPer = 3.25):
        
        self.death = Model( polynomial(DeathModel) )
        self.start_weight = StartWeight
        self.barn_size = BarnSize
        self.death_loss = DeathLossPer

        self.adjust_death

        self.pig = PigModel
        self.sales = SalesModel

        self.set_awg
        self.set_g_total
        self.set_g_cum

        self.set_awfi
        self.set_fi_total
        self.set_fi_cum

        self.set_awfc
        self.set_fc_cum

        self.aw_feed_cost = Model(lambda x: self.pig.aw_feed_cost.model(x) * self.alive.model(x))

    @property 
    def calc_sales(self):
        wk = self.pig.awg.calc_week(self.sales.live_avg - self.start_weight)
        self.revenue_total = self.sales.revenue_avg * self.alive.model(wk)
        self.feed_total = self.aw_feed_cost.integrate(0, wk)
        self.revenue_net = self.revenue_total - self.feed_total

    def calc_feed_cost_diff(self, lb, ub, data_type = "wk"):
        if data_type == "wt":
            lb = self.pig.awg.calc_week(lb - self.start_weight)
            ub = self.pig.awg.calc_week(ub - self.start_weight)

        return self.aw_feed_cost.integrate(lb, ub)

    @property 
    def set_awg(self):
        p = self.pig.awg.model * self.alive.model
        self.awg = Model( polynomial(p.coef) )

    @property 
    def set_g_total(self):
        p = self.awg.model.integ()
        self.g_total = Model( polynomial(p.coef) )

    @property 
    def set_g_cum(self):
        p = self.awg.model.integ()
        m = polynomial(p.coef)
        self.g_cum = Model( lambda x: m(x) / (x * 7 * self.barn_size) )

    @property 
    def set_awfc(self):
        p = self.pig.awfc.model * self.alive.model
        self.awfc = Model( polynomial(p.coef) )

    @property 
    def set_fc_cum(self):
        self.fc_cum = Model(lambda x: self.fi_cum.model(x) / self.g_cum.model(x))

    @property 
    def set_awfi(self):
        p = self.pig.awfi.model * self.alive.model
        self.awfi = Model( polynomial(p.coef) )

    @property 
    def set_fi_total(self):
        p = self.awfi.model.integ()
        self.fi_total = Model( polynomial(p.coef) )

    @property 
    def set_fi_cum(self):
        p = self.awfi.model.integ()
        m = polynomial(p.coef)
        self.fi_cum = Model( lambda x: m(x) / (x * 7 * self.barn_size) )

    @property
    def adjust_death(self):
        adjust = self.death.integrate(0, 26)
        p = self.death.model * (self.death_loss / 100 * self.barn_size / adjust)
        self.death = Model( polynomial(p.coef) )
        self.set_alive

    @property
    def set_alive(self):
        p = self.barn_size - self.death.model.integ()
        self.alive = Model( polynomial(p.coef) )

class Model():
    def __init__(self, func):
        self.model = func

    def integrate(self, lb = 0, ub = 26, div = 1, x = None, coef = 1): 
        if x is not None:
            self.model.coef[coef] = x

        return quad(self.model, lb, ub, limit=1000)[0] / div

    def calc_week(self, zero, lb = 0):
        return newton(lambda x: self.integrate(lb, x) - zero, 26)