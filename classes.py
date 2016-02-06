CARGILL_WB = "C:\Users\Jonathan.WSF\Documents\GitHub\walk-stock-farm\Cargill Lean Value Matrix.csv"
WT_CUTOFF_DEFAULT = [50, 80, 120, 160, 200, 225, 245, 265]

import math
import multiprocessing as mp
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
        
        self.carcass_avg = numpy.array([CarcassAvg])
        self.carcass_std_dev = CarcassStdDev

        self.lean_avg = LeanAvg
        self.lean_std_dev = LeanStdDev
        self.yield_avg = YieldAvg
        
        self.base_price = BasePrice

        self.packer = Packer
        self.lean_dist = LeanDist
        self.carcass_dist = CarcassDist

        self.calc_matrix_factor
        self.calc_revenue

    @property 
    def live_avg(self):
        return self.carcass_avg / (self.yield_avg / 100)

    @property
    def carcass_s(self):
        return math.sqrt(3 * self.carcass_std_dev ** 2 / math.pi ** 2)

    @property 
    def calc_matrix_factor(self):
        if self.lean_dist == "norm":
            prob_dist = norm
            std_dev = self.lean_std_dev

        if self.packer == "Cargill":
            self.packer_wt_arr_lb = numpy.array([-numpy.inf, 0.5, 140.5, 147.5, 154.5, 162.5, 169.5, 176.5, 184.5, 191.5, 198.5, 206.5, 213.5, 221.5, 228.5, 235.5, 242.5, 244.5, 249.5, 256.5, 263.5])
            self.packer_wt_arr_ub = numpy.array([0.5, 140.5, 147.5, 154.5, 162.5, 169.5, 176.5, 184.5, 191.5, 198.5, 206.5, 213.5, 221.5, 228.5, 235.5, 242.5, 244.5, 249.5, 256.5, 263.5, numpy.inf])
            self.packer_lean_arr_lb = numpy.array([-numpy.inf, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5])
            self.packer_lean_arr_ub = numpy.array([39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, numpy.inf])
            
            self.lean_dist_arr = prob_dist.cdf(self.packer_lean_arr_ub, self.lean_avg, std_dev) - prob_dist.cdf(self.packer_lean_arr_lb, self.lean_avg, std_dev)

            self.packer_matrix_df = pandas.read_csv(CARGILL_WB, header=None)

            self.matrix_factor = self.packer_matrix_df.multiply(self.lean_dist_arr, 'index').sum(0)

    @property 
    def calc_revenue(self):        
        if self.carcass_dist == "norm":
            prob_dist = norm
            std_dev = self.carcass_std_dev
        elif self.carcass_dist == "logistic":
            prob_dist = logistic
            std_dev = self.carcass_s

        self.wt_dist_df = pandas.DataFrame(index = range(len(self.matrix_factor)), columns = range(len(self.carcass_avg)))
        for x in range(len(self.carcass_avg)):
            self.wt_dist_df[x] = (prob_dist.cdf(self.packer_wt_arr_ub, self.carcass_avg[x], std_dev) - prob_dist.cdf(self.packer_wt_arr_lb, self.carcass_avg[x], std_dev)) / 100

        self.base_price_adj = self.wt_dist_df.multiply(self.matrix_factor + self.base_price, 'index').sum(0)
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

        self.set_feed_cost
        self.set_aw_feed_cost
        self.set_feed_cost_total
        self.set_feed_cost_cum

    def shift_awg(self, zero, lb, ub):
        self.awg = Model( self.awg.model * newton(lambda x: quad(self.awg.model * x, lb, ub)[0] - zero, 1))

    def shift_awfc(self, zero, lb, ub):
        self.awfc = Model( polynomial( [self.awfc.model.coef[0], newton(lambda x: self.opt_fc(x, ub) - zero, 0)] ) )

    def opt_fc(self, x, wk):
        self.awfc.model.coef[1] = x
        self.set_awfi
        self.set_fi_total
        self.set_fi_cum
        self.set_fc_cum
        return self.fc_cum.model(wk)

    @property
    def set_g_total(self):
        self.g_total = Model( self.awg.model.integ() )

    @property 
    def set_g_cum(self):
        self.g_cum = Model(lambda x: self.g_total.model(x) / (x * 7) )

    @property
    def set_fc_cum(self):
        self.fc_cum = Model(lambda x: self.fi_cum.model(x) / self.g_cum.model(x))

    @property
    def set_awfi(self):
        self.awfi = Model( self.awg.model * self.awfc.model )

    @property
    def set_fi_total(self):
        self.fi_total = Model( self.awfi.model.integ() ) 

    @property 
    def set_fi_cum(self):
        self.fi_cum = Model(lambda x: self.fi_total.model(x) / (x * 7) )

    @property 
    def set_feed_cost(self):
        self.feed_cost = Model(lambda x: self.heaviside_combined(x))
    
    @property 
    def set_aw_feed_cost(self):
        self.aw_feed_cost = Model(lambda x: self.feed_cost.model(x) * self.awfi.model(x))

    @property 
    def set_feed_cost_total(self):
        self.feed_cost_total = Model(lambda x: self.feed_cost.model(x) * self.fi_total.model(x))

    @property 
    def set_feed_cost_cum(self):
        self.feed_cost_cum = Model(lambda x: self.feed_cost_total.model(x) - self.feed_cost_total.model(0))

    @property
    def calc_week_cutoff(self):
        self.cutoff_week = self.awg.roots(self.cutoff_wt - self.start_weight)

    def heaviside(self, x):
        return 0.5 + 0.5 * numpy.tanh(5000 * x)

    def heaviside_combined(self, x):
        # return numpy.sum( (self.cutoff_price_ub - self.cutoff_price_lb) * self.heaviside(x - self.cutoff_week) ) 
        func = self.cutoff_price[0] * self.heaviside(x)
        for i in range(1, len(self.cutoff_week), 1):
            func = func + (self.cutoff_price[i] - self.cutoff_price[i - 1]) * self.heaviside(x - self.cutoff_week[i])  

        return func 

class BarnModel():
    def __init__(self, DeathModel, PigModel, SalesModel, StartWeight = 12, BarnSize = 2500, PigSpaces = 2400, WeeklyRent = 2030, DeathLossPer = 3.25, DiscountLossPer = 0.75, ModelType = "barn"):
        
        self.death = Model( polynomial(DeathModel) )
        self.start_weight = StartWeight
        self.barn_size = BarnSize
        self.death_loss = DeathLossPer
        self.discount_loss = DiscountLossPer
        self.model_type = ModelType
        self.rent_week = WeeklyRent
        self.pig_spaces = PigSpaces

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

        self.set_aw_feed_cost
        self.set_feed_cost_total
        self.set_feed_cost_cum

        self.calc_revenue_base

    def calc_rev_curve(self, arr, arr_type = "live_wt", PerPig = False):

        if arr_type == "live_wt":
            self.sales.carcass_avg = arr * (self.sales.yield_avg / 100)
            self.sales.calc_revenue
            self.calc_sales
            rev_arr = self.revenue_net - self.revenue_base

        return rev_arr

    @property 
    def calc_revenue_base(self):
        self.sales.carcass_avg = numpy.array([250 * (self.sales.yield_avg / 100)])
        self.sales.calc_revenue
        self.calc_sales
        self.revenue_base = self.revenue_net[0]

    def calc_opt_price_curve(self, arr, PerPig = False):
        opt_price_arr = []
        wt_arr = numpy.arange(260, 325, 5)

        for i in range(0, len(arr)):
            self.sales.base_price = arr[i]
            opt_price_arr.append( wt_arr[ numpy.argmax( self.calc_rev_curve(wt_arr, PerPig = PerPig) ) ] )

        return opt_price_arr

    @property 
    def calc_sales(self):
        wk = self.pig.awg.roots(self.sales.live_avg - self.start_weight)

        self.revenue_total = self.sales.revenue_avg * self.alive.model(wk)
        self.feed_total = self.feed_cost_cum.model(wk)
        self.rent_total = self.rent_week * wk
        if self.model_type == "cut":
            self.rent_total = self.rent_total * (self.alive.model(wk) / self.pig_spaces)

        self.revenue_net = self.revenue_total - self.feed_total - self.rent_total
        self.revenue_net_pig = self.revenue_net / self.alive.model(wk)

    def calc_feed_cost_diff(self, lb, ub, data_type = "wk"):
        if data_type == "wt":
            lb = self.pig.awg.calc_week(lb - self.start_weight)
            ub = self.pig.awg.calc_week(ub - self.start_weight)

        return self.aw_feed_cost.integrate(lb, ub)

    @property 
    def set_awg(self):
        self.awg = Model( self.pig.awg.model * self.alive.model )

    @property 
    def set_g_total(self):
        self.g_total = Model( self.awg.model.integ() )

    @property 
    def set_g_cum(self):
        self.g_cum = Model(lambda x: self.g_total.model(x) / (x * 7 * self.barn_size))

    @property 
    def set_awfc(self):
        self.awfc = Model( self.pig.awfc.model * self.alive.model )

    @property 
    def set_fc_cum(self):
        self.fc_cum = Model(lambda x: self.fi_cum.model(x) / self.g_cum.model(x))

    @property 
    def set_awfi(self):
        self.awfi = Model( self.pig.awfi.model * self.alive.model )

    @property 
    def set_fi_total(self):
        self.fi_total = Model( self.awfi.model.integ() )

    @property 
    def set_fi_cum(self):
        self.fi_cum = Model(lambda x: self.fi_total.model(x) / (x * 7 * self.barn_size))

    @property 
    def set_aw_feed_cost(self):
        self.aw_feed_cost = Model(lambda x: self.pig.feed_cost.model(x) * self.awfi.model(x) * self.alive.model(x))

    @property 
    def set_feed_cost_total(self):
        self.feed_cost_total = Model(lambda x: self.pig.feed_cost.model(x) * self.fi_total.model(x))

    @property 
    def set_feed_cost_cum(self):
        self.feed_cost_cum = Model(lambda x: self.feed_cost_total.model(x) - self.feed_cost_total.model(0))

    @property
    def adjust_death(self):
        self.set_death_total
        self.death = Model( self.death.model * (self.death_loss / 100 * self.barn_size / self.death_total.diff(26)) )
        self.set_death_total
        self.set_death_cum
        self.set_alive

    @property 
    def set_death_total(self):
        self.death_total = Model( self.death.model.integ() )

    @property 
    def set_death_cum(self):
        self.death_cum = Model( lambda x: self.death_total(x) - self.death_total(0))

    @property
    def set_alive(self):
        self.alive = Model( self.barn_size - self.death.model.integ() )

class Model():
    def __init__(self, func):
        self.model = func

    def integrate(self, lb = 0, ub = 26, div = 1, x = None, coef = 1): 
        if x is not None:
            self.model.coef[coef] = x

        return quad(self.model, lb, ub, limit=1000)[0] / div

    def diff(self, ub, lb = 0):
        return self.model(ub) - self.model(lb)

    def calc_week(self, zero, lb = 0):
        return newton(lambda x: self.integrate(lb, x) - zero, 26)

    def roots(self, zero):
        weeks = numpy.array(zero, numpy.float32)

        for i in range(len(zero)):
            p = zero[i] - self.model.integ()
            weeks[i] = numpy.max( numpy.real( filter( lambda x: not numpy.iscomplex(x), p.roots() ) ) )

        return weeks