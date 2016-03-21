CARGILL_WB = "C:\Users\Jonathan.WSF\Documents\GitHub\walk-stock-farm\Cargill Lean Value Matrix.csv"
# CARGILL_WB = "FlaskWebProject/static/storage/Cargill Lean Value Matrix.csv"
WT_CUTOFF_DEFAULT = [50, 80, 120, 160, 200, 225, 245, 265]

import operator
import math
import multiprocessing as mp
import numpy
import pandas
from scipy.stats import norm
from scipy.stats import logistic

from scipy.integrate import quad
from scipy.optimize import newton
from scipy.optimize import fsolve

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
    def __init__(self, awgModel = [0], awfcModel = [0], AvgWeeksInBarn = 24.5,
                 awgAdjust = None, awfcAdjust = None, 
                 PriceCutoff = [0], WtCutoff = WT_CUTOFF_DEFAULT, WeekCutoff = None,
                 StartWeight = 12):

        self.start_weight = StartWeight
        self.avg_weeks_in_barn = AvgWeeksInBarn

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
    def __init__(self, DeathModel, PigModel, SalesModel, StartNum = 2500, PigSpaces = 2400, WeeklyRent = 2030, DeathLossPer = 3.25, DiscountLossPer = 0.75, ModelType = "barn"):
        
        self.pig = PigModel
        self.sales = SalesModel

        self.death = Model( polynomial(DeathModel) )
        self.start_num = StartNum
        self.death_loss = DeathLossPer
        self.discount_loss = DiscountLossPer
        self.model_type = ModelType
        self.rent_week = WeeklyRent
        self.pig_spaces = PigSpaces

        self.adjust_death

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
            # rev_arr = self.revenue_net - self.revenue_base
            rev_arr = self.revenue_net

        return rev_arr

    @property 
    def calc_revenue_base(self):
        self.sales.carcass_avg = numpy.array([250 * (self.sales.yield_avg / 100)])
        self.sales.calc_revenue
        self.calc_sales
        self.revenue_base = self.revenue_net[0]

    def calc_opt_price_curve(self, arr, PerPig = False):
        opt_price_arr = []
        wt_arr = numpy.arange(270, 325, 5)

        for i in range(0, len(arr)):
            self.sales.base_price = arr[i]
            opt_price_arr.append( wt_arr[ numpy.argmax( self.calc_rev_curve(wt_arr, PerPig = PerPig) ) ] )

        return opt_price_arr

    @property 
    def calc_sales(self):
        wk = self.pig.awg.roots(self.sales.live_avg - self.pig.start_weight)

        self.revenue_total = self.sales.revenue_avg * self.alive.model(wk)
        self.feed_total = self.feed_cost_cum.model(wk)
        self.rent_total = self.rent_week * wk
        if self.model_type == "cut":
            self.rent_total = self.rent_total * (self.alive.model(wk) / self.pig_spaces)

        self.revenue_net = self.revenue_total - self.feed_total - self.rent_total
        self.revenue_net_pig = self.revenue_net / self.alive.model(wk)

    def calc_feed_cost_diff(self, lb, ub, data_type = "wk"):
        if data_type == "wt":
            lb = self.pig.awg.calc_week(lb - self.pig.start_weight)
            ub = self.pig.awg.calc_week(ub - self.pig.start_weight)

        return self.aw_feed_cost.integrate(lb, ub)

    @property 
    def set_awg(self):
        self.awg = Model( self.pig.awg.model * self.alive.model )

    @property 
    def set_g_total(self):
        self.g_total = Model( self.awg.model.integ() )

    @property 
    def set_g_cum(self):
        self.g_cum = Model(lambda x: self.g_total.model(x) / (x * 7 * self.start_num))

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
        self.fi_cum = Model(lambda x: self.fi_total.model(x) / (x * 7 * self.start_num))

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
        self.death = Model( self.death.model * (self.death_loss / 100 * self.start_num / self.death_total.diff(self.pig.avg_weeks_in_barn)) )
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
        self.alive = Model( self.start_num - self.death.model.integ() )

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

        # p_int = p.integ()
        # p_roots = (p_int - zero).roots()
        # print numpy.real(p_roots[numpy.searchsorted(p_roots, 0.1)])

        for i in range(len(zero)):
            p = zero[i] - self.model.integ()
            weeks[i] = numpy.max( numpy.real( filter( lambda x: not numpy.iscomplex(x), p.roots() ) ) )

        return weeks

class BarnModelPiece():
    def __init__(self, DeathModel, RentModel, PigModel, SalesModel, StartNum = 2500, DeathLossPer = 3.25, DiscountLossPer = 0.75, DiscountPricePer = 50, AvgWeeksInBarn = 24.5):
        
        self.pig = PigModel
        self.sales = SalesModel
        self.death = DeathModel
        self.rent = RentModel

        self.start_num = StartNum
        self.death_loss = DeathLossPer
        self.discount_loss = DiscountLossPer
        self.discount_num = self.start_num * (self.discount_loss / 100)
        self.discount_price_per = DiscountPricePer
        self.avg_weeks_in_barn = AvgWeeksInBarn

        self.adjust_death
        self.set_curves

        # self.calc_revenue_base

    @property
    def set_curves(self):
        self.set_awg
        self.set_g_total

        self.set_awfc

        self.set_awfi
        self.set_fi_total

        self.set_aw_feed_cost
        self.set_feed_cost_total

        self.set_rent_total

    def calc_rev_curve(self, arr, arr_type = "live_wt", PerPig = False):

        if arr_type == "live_wt":
            self.sales.carcass_avg = arr * (self.sales.yield_avg / 100)
            self.sales.calc_revenue
            self.calc_sales
            # rev_arr = self.revenue_net - self.revenue_base
            rev_arr = self.revenue_net

        return rev_arr

    @property 
    def calc_revenue_base(self):
        self.sales.carcass_avg = numpy.array([250 * (self.sales.yield_avg / 100)])
        self.sales.calc_revenue
        self.calc_sales
        self.revenue_base = self.revenue_net[0]

    def calc_opt_price_curve(self, arr, PerPig = False):
        opt_price_arr = []
        wt_arr = numpy.arange(270, 325, 5)

        for i in range(0, len(arr)):
            self.sales.base_price = arr[i]
            opt_price_arr.append( wt_arr[ numpy.argmax( self.calc_rev_curve(wt_arr, PerPig = PerPig) ) ] )

        return opt_price_arr

    @property 
    def calc_sales(self):
        wk = self.pig.wt_total.weeks(self.sales.live_avg)

        self.revenue_total = (self.sales.revenue_avg * self.market_total(wk)) + (self.sales.revenue_avg * (self.discount_price_per / 100) * self.discount_num)
        # if self.model_type == "cut":
        #     self.rent_total = self.rent_total * (self.alive_total(wk) / self.pig_spaces)

        self.revenue_net = self.revenue_total - self.feed_cost_total(wk) - self.rent_total(wk)
        self.revenue_net_pig = self.revenue_net / self.alive_total(wk)

    @property 
    def set_awg(self):
        self.awg = self.pig.awg * self.alive_total

    @property 
    def set_g_total(self):
        self.g_total = self.awg.integ()

    def g_cum(self):
        self.g_cum = self.g_total(x) / (x * 7 * self.alive_total(x))

    @property 
    def set_awfc(self):
        self.awfc = self.pig.awfc * self.alive_total

    def fc_cum(self):
        return self.fi_cum(x) / self.g_cum(x)

    @property 
    def set_awfi(self):
        self.awfi = self.pig.awfi * self.alive_total

    @property 
    def set_fi_total(self):
        self.fi_total = self.awfi.integ()

    def fi_cum(self):
        return self.fi_total(x) / (x * 7 * self.alive_total(x))

    @property 
    def set_aw_feed_cost(self):
        self.aw_feed_cost = self.pig.feed_cost * self.awfi

    @property 
    def set_feed_cost_total(self):
        self.feed_cost_total = self.aw_feed_cost.integ()

    @property
    def adjust_death(self):
        self.death = self.death * ( ((self.death_loss / 100) * self.start_num) / self.death.integ()(self.avg_weeks_in_barn) )
        self.set_death_total
        self.set_alive_total
        self.set_market_total

    @property 
    def set_death_total(self):
        self.death_total = self.death.integ()

    @property
    def set_alive_total(self):
        self.alive_total = self.start_num - self.death_total

    @property
    def set_market_total(self):
        self.market_total = self.alive_total - ( self.start_num * (self.discount_loss / 100) )

    @property
    def set_rent_total(self):
        self.rent_total = self.rent.integ()

class PigGrowthModelPiece():
    def __init__(self, awgModel, awfcModel, feedCostModel, awgAdjust = None, awfcAdjust = None, StartWeight = 12):
        self.start_weight = StartWeight

        self.awg = awgModel
        self.awfc = awfcModel
        self.feed_cost = feedCostModel

        if awgAdjust is not None:
            self.adjust_awg(awgAdjust[0], awgAdjust[1], awgAdjust[2])

        self.g_total = self.awg.integ()
        self.set_feed

        if awfcAdjust is not None:
            self.adjust_awfc(awfcAdjust[0], awfcAdjust[1])

        self.shift_curves

    def opt_awg(self, x, week_start, week_end):
        p_len = len(self.awg.poly_arr) - 1

        eq = []
        eq.append(x[0]*self.awg.poly_arr[0].integ()(x[1]) - self.awg.poly_arr[0].integ()(week_start) - (self.awg.break_arr_wt[0] - 12))

        for i in range(1, p_len):
            eq.append(x[0] * self.awg.poly_arr[i].integ()(x[i + 1]) - x[0] * self.awg.poly_arr[i].integ()(x[i]) - (self.awg.break_arr_wt[i] - self.awg.break_arr_wt[i - 1]))

        eq.append(x[0] * self.awg.poly_arr[p_len].integ()(week_end) - x[0] * self.awg.poly_arr[p_len].integ()(x[p_len]) - (self.awg.break_arr_wt[p_len - 1] - self.awg.break_arr_wt[p_len - 2]))
        return eq

    def adjust_awg(self, zero, week_start = 0, week_end = 24.5):
        x0_arr = numpy.array([1])
        x0_arr = numpy.append(x0_arr, self.awg.break_arr)
        f = fsolve(self.opt_awg, x0_arr, args=(week_start, week_end))
        self.awg = self.awg * f[0]
        self.awg.break_arr = numpy.array(f[1:])
        self.awfc.break_arr = numpy.array(f[1:])

    def adjust_awfc(self, zero, week_end = 24.5):
        self.awfc = self.awfc * (zero / self.fc_cum(week_end))

    @property
    def shift_curves(self):
        if self.start_weight != 12:
            self.awg.shift_x = self.g_total.weeks(self.start_weight - 12)
            self.awfc.shift_x = self.g_total.weeks(self.start_weight - 12)

        self.set_gain
        self.set_feed
        self.feed_cost.break_arr = self.wt_total.weeks(self.feed_cost.break_arr_wt)
        self.set_feed_cost

    @property
    def set_gain(self):
        self.g_total = self.awg.integ() - (self.start_weight - 12)
        self.wt_total = self.g_total + self.start_weight

    @property
    def set_feed(self):
        self.awfi = self.awg * self.awfc
        self.fi_total = self.awfi.integ()

    @property
    def set_feed_cost(self):
        self.aw_feed_cost = self.awfi * self.feed_cost
        self.feed_cost_total = self.aw_feed_cost.integ()

    def fi_cum(self, x):
        return self.fi_total(x) / (7 * x)

    def g_cum(self, x):
        return self.g_total(x) / (7 * x)

    def fc_cum(self, x):
        return self.fi_cum(x) / self.g_cum(x)

class PiecePolynomial():
    def __init__(self, polyArr = [polynomial([0, 1]), polynomial([0, 2])], breakArr = [1], breakArrWt = [1], horizShift = 0):
        self.shift_x = horizShift

        if isinstance(polyArr, (list, set, tuple,)):
            self.poly_arr = numpy.array(polyArr)

        if isinstance(breakArr, (list, set, tuple,)):
            self.break_arr = numpy.array(breakArr)
        elif not isinstance(breakArr, numpy.ndarray):
            self.break_arr = numpy.array([breakArr])
        else:
            self.break_arr = breakArr

        if isinstance(breakArrWt, (list, set, tuple,)):
            self.break_arr_wt = numpy.array(breakArrWt)
        elif not isinstance(breakArrWt, numpy.ndarray):
            self.break_arr_wt = numpy.array([breakArrWt])
        else:
            self.break_arr_wt = breakArrWt

    def __call__(self, x):
        if isinstance(x, (list, set, tuple,)):
            x = numpy.array(x)
        elif not isinstance(x, numpy.ndarray):
            x = numpy.array([x])

        x = x + self.shift_x

        i_arr = self.arr_index(x)
        u_arr = numpy.unique(i_arr, return_index=True)
        b_arr = u_arr[1]
        p_arr = u_arr[0]

        if len(p_arr) == 1:
            arr = self.poly_arr[p_arr[0]](x)
        else:
            arr = numpy.array([])
            for i in range(0, len(p_arr) - 1):
                arr = numpy.append(arr, self.poly_arr[p_arr[i]](x[b_arr[i]:b_arr[i + 1]]))
            arr = numpy.append(arr, self.poly_arr[p_arr[len(p_arr) - 1]](x[b_arr[len(b_arr) - 1]:]))

        return arr

    def arr_index(self, x):
        return self.break_arr.searchsorted(x, "right")

    def math_operation(self, x, op_char, r_bool = True):
        if op_char == "add":
            op_func = operator.add
        elif op_char == "sub":
            op_func = operator.sub
        elif op_char == "mul":
            op_func = operator.mul
        elif op_char == "div":
            op_func = operator.div

        if isinstance(x, PiecePolynomial):
            i = 0
            j = 0
            k = 0

            if r_bool:
                a = self.poly_arr
                b = x.poly_arr
            else:
                a = x.poly_arr
                b = self.poly_arr

            len_self = len(self.break_arr)
            len_x = len(x.break_arr)
            len_comb = len_self + len_x - len(numpy.intersect1d(self.break_arr, x.break_arr))

            p_arr = []
            b_arr = []

            while k < len_comb and i < len_self and j < len_x:
                p_arr.append(op_func(self.poly_arr[i], x.poly_arr[j]))

                if self.break_arr[i] > x.break_arr[j]:
                    b_arr.append(x.break_arr[j])
                    j = j + 1

                elif self.break_arr[i] < x.break_arr[j]:
                    b_arr.append(self.break_arr[i])
                    i = i + 1

                else:
                    b_arr.append(x.break_arr[j])
                    i = i + 1
                    j = j + 1
                
                k = k + 1

            if i < len_self:
                while k < len_comb and i < len_self:
                    p_arr.append(op_func(self.poly_arr[i], x.poly_arr[j]))
                    b_arr.append(self.break_arr[i])

                    i = i + 1
                    k = k + 1
            elif j < len_x:
                while k < len_comb and j < len_x:
                    p_arr.append(op_func(self.poly_arr[i], x.poly_arr[j]))
                    b_arr.append(x.break_arr[j])

                    j = j + 1
                    k = k + 1

            p_arr.append(op_func(self.poly_arr[len(self.poly_arr) - 1], x.poly_arr[len(x.poly_arr) - 1]))

            return PiecePolynomial(p_arr, b_arr, self.break_arr_wt, self.shift_x)

        else:
            p_arr = []
            for poly in self.poly_arr:
                p_arr.append(op_func(poly, x))

        return PiecePolynomial(p_arr, self.break_arr, self.break_arr_wt, self.shift_x)

    def math_operation_r(self, x, op_char):
        if op_char == "add":
            op_func = operator.add
        elif op_char == "sub":
            op_func = operator.sub
        elif op_char == "mul":
            op_func = operator.mul
        elif op_char == "div":
            op_func = operator.div

        if isinstance(x, PiecePolynomial):
            i = 0
            j = 0
            k = 0

            len_self = len(self.break_arr)
            len_x = len(x.break_arr)
            len_comb = len_self + len_x - len(numpy.intersect1d(self.break_arr, x.break_arr))

            p_arr = []
            b_arr = []

            while k < len_comb and i < len_self and j < len_x:
                p_arr.append(op_func(x.poly_arr[j], self.poly_arr[i]))

                if self.break_arr[i] > x.break_arr[j]:
                    b_arr.append(x.break_arr[j])
                    j = j + 1

                elif self.break_arr[i] < x.break_arr[j]:
                    b_arr.append(self.break_arr[j])
                    i = i + 1

                else:
                    b_arr.append(x.break_arr[j])
                    i = i + 1
                    j = j + 1
                
                k = k + 1

            if i < len_self:
                while k < len_comb and i < len_self:
                    p_arr.append(op_func(x.poly_arr[j], self.poly_arr[i]))
                    b_arr.append(self.break_arr[j])

                    i = i + 1
                    k = k + 1
            elif j < len_x:
                while k < len_comb and j < len_x:
                    p_arr.append(op_func(x.poly_arr[j], self.poly_arr[i]))
                    b_arr.append(x.break_arr[j])

                    j = j + 1
                    k = k + 1

            p_arr.append(op_func(x.poly_arr[len(x.poly_arr) - 1], self.poly_arr[len(self.poly_arr) - 1]))
            # print p_arr
            # print b_arr

            return PiecePolynomial(p_arr, b_arr, self.break_arr_wt, self.shift_x)

        else:
            p_arr = []
            for poly in self.poly_arr:
                p_arr.append(op_func(x, poly))

        return PiecePolynomial(p_arr, self.break_arr, self.break_arr_wt, self.shift_x)

    def __mul__(self, x):
        return self.math_operation(x, "mul")

    def __rmul__(self, x):
        return self.math_operation_r(x, "mul")

    def __div__(self, x):
        return self.math_operation(x, "div")

    def __rdiv__(self, x):
        return self.math_operation_r(x, "div")

    def __add__(self, x):
        return self.math_operation(x, "add")

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return self.math_operation(x, "sub")

    def __rsub__(self, x):
        return self.math_operation_r(x, "sub")

    def deriv(self):
        p_arr = []

        for poly in self.poly_arr:
            p_arr.append(poly.deriv())

        return PiecePolynomial(p_arr, self.break_arr, self.break_arr_wt, self.shift_x)

    def integ(self):
        p_arr = [self.poly_arr[0].integ()]
        for i in range(0, len(self.break_arr)):
            if self.poly_arr[i + 1](self.break_arr[i]) == p_arr[i](self.break_arr[i]):
                p_arr.append(self.poly_arr[i + 1].integ())
            else:
                p_arr.append(self.poly_arr[i + 1].integ() + (p_arr[i](self.break_arr[i]) - self.poly_arr[i + 1].integ()(self.break_arr[i])))

        return PiecePolynomial(p_arr, self.break_arr, self.break_arr_wt, self.shift_x)

    def roots(self):
        r_arr = numpy.array([])

        for poly in self.poly_arr:
            r_arr = numpy.append(r_arr, poly.roots())

        return r_arr

    def root(self):
        r_arr = numpy.array([])

        for poly in self.poly_arr:
            p_roots = poly.roots()
            r_arr = numpy.append(r_arr,  numpy.real(p_roots[numpy.searchsorted(p_roots, 0.001)]))

        return r_arr        

    def weeks(self, x):
        if isinstance(x, (list, set, tuple,)):
            x = numpy.array(x)
        elif not isinstance(x, numpy.ndarray):
            x = numpy.array([x])

        roots = numpy.array([])
        for zero in x:
            p_roots = (self - zero).root()
            
            break_bool = False
            for b in self.break_arr:
                for root in p_roots:
                    if root < b:
                        roots = numpy.append(roots, root)
                        break_bool = True
                        break

                if break_bool:
                    break

            if not break_bool:
                roots = numpy.append(roots, root)

        return roots

    # def adjust_model(self, zero, week_start = 0, week_end = 24.5):
    #     p_int = self.integ()
    #     print zero
    #     print p_int(week_end)
    #     print p_int(week_start)

    #     self = self * (zero / (p_int(week_end) - p_int(week_start)))

    #     if len(self.break_arr_wt) > 0:
    #         for i in range(0, len(self.break_arr_wt)):
    #             p_roots = (p_int.poly_arr[i] - self.break_arr_wt[i] + 12).roots()
    #             self.break_arr[i] = numpy.real(p_roots[numpy.searchsorted(p_roots, 0.001)])

