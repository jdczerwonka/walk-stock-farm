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
    def __init__(self, MarketSize = 2400, CarcassAvg = 220, CarcassStdDev = 18,
                 LeanAvg = 54, LeanStdDev = 2.1, YieldAvg = 76,
                 BasePrice = 80, FeedPrice = 0.10, BaseLiveWt = 250, DeathLoss = 3.25,
                 Packer = "Cargill", LeanDist = "norm", CarcassDist = "logistic"):
        
        self.market_size = MarketSize
        
        self.carcass_avg = CarcassAvg
        self.carcass_std_dev = CarcassStdDev

        self.lean_avg = LeanAvg
        self.lean_std_dev = LeanStdDev
        self.yield_avg = YieldAvg
        
        self.base_price = BasePrice
        self.feed_price = FeedPrice
        self.base_live_wt = BaseLiveWt
        self.death_loss = DeathLoss

        self.packer = Packer
        self.lean_dist = LeanDist
        self.carcass_dist = CarcassDist

        self.initialize_packer()
        self.initialize_lean_dist()
        
        self.growth_model = GrowthModel()

    @property
    def carcass_s(self):
        return math.sqrt(3 * self.carcass_std_dev ** 2 / math.pi ** 2)

    @property
    def adjust_death_model(self):
        adjust = self.growth_model.death.integrate(0, 26)
        self.growth_model.death.model.coef = self.growth_model.death.model.coef * (self.death_loss / 100 * self.market_size / adjust)

    def initialize_packer(self):
        if self.packer == "Cargill":
            self.lean_arr_lb = -0.5
            self.lean_arr_lb = numpy.append(self.lean_arr_lb, numpy.arange(40.5, 63.5))

            self.lean_arr_ub = numpy.arange(40.5, 63.5)
            self.lean_arr_ub = numpy.append(self.lean_arr_ub, 100.5)

            self.packer_wt_arr = [0, 141, 148, 155, 163, 170, 177, 185, 192, 199, 207, 214, 222, 229, 236, 243, 245, 250, 257, 264, 999]
            self.packer_matrix_df = pandas.read_csv(CARGILL_WB, header=None)

    def initialize_lean_dist(self):
        if self.lean_dist == "norm":
            self.prob_arr = numpy.round(norm.cdf(self.lean_arr_ub, self.lean_avg, self.lean_std_dev) - norm.cdf(self.lean_arr_lb, self.lean_avg, self.lean_std_dev), 4)

    def calculate_range(self, arr, factor):
        data_table = pandas.DataFrame(columns=(factor, "rev_net", "rev_net_pig"))
        
        if factor == "live_avg":
            for wt in arr:
                self.carcass_avg = wt * ( self.yield_avg / 100 )
                self.calculate_model()
                data_table.loc[len(data_table) + 1] = [wt, self.rev_net, self.rev_net_pig]
        elif factor == "carcass_std_dev":
            for std_dev in arr:
                self.carcass_std_dev = std_dev
                self.calculate_model
                data_table.loc[len(data_table) + 1] = [wt, self.rev_net, self.rev_net_pig]
                
        return data_table

    def calculate_model(self):
        self.calculate_rev()
        self.calculate_feed()
        self.calculate_death()

        self.rev_net = self.sum("rev_total") - self.sum("excess_feed_total_cost") - self.death_cost
        self.rev_net_pig = self.rev_net / self.market_size_adj

    def calculate_rev(self):        
        CarcassL = numpy.arange(0.5, 399.5)
        CarcassU = numpy.arange(1.5, 400.5)
        
        if self.carcass_dist == "logistic":
            Num = numpy.round((logistic.cdf(CarcassU, self.carcass_avg, self.carcass_s) -
                logistic.cdf(CarcassL, self.carcass_avg, self.carcass_s)) * self.market_size, 0)
        elif self.carcass_dist == "norm":
            Num = numpy.round((norm.cdf(CarcassU, self.carcass_avg, self.carcass_std_dev) -
                norm.cdf(CarcassL, self.carcass_avg, self.carcass_std_dev)) * self.market_size, 0)
            
        CarcassDF = pandas.DataFrame({  "carcass_avg" : numpy.arange(1, 400),
                                        "num" : Num
                                    })
                              
        CarcassDF = CarcassDF[CarcassDF["num"] > 0]

        CarcassDF["matrix_factor"] = self.calculate_matrix_factor(CarcassDF["carcass_avg"])
        CarcassDF["rev_head"] = (CarcassDF["carcass_avg"] / 100) * (CarcassDF["matrix_factor"] + self.base_price)
        CarcassDF["rev_total"] = CarcassDF["rev_head"] * CarcassDF["num"]

        self.carcass_df = CarcassDF

    def calculate_matrix_factor(self, CarcassWt):
        MatrixFactor = [0]
        for index, val in CarcassWt.iteritems():
            x = 0
            while self.packer_wt_arr[x] < val:
                x = x + 1

            MatrixFactor = numpy.append(MatrixFactor, numpy.sum(self.packer_matrix_df[x - 1] * self.prob_arr))  

        MatrixFactor = numpy.delete(MatrixFactor, 0)
        
        return MatrixFactor

    def calculate_feed(self):
        self.carcass_df["live_avg"] = self.carcass_df["carcass_avg"] / ( self.yield_avg / 100 )
        self.carcass_df["excess_gain"] = self.carcass_df["live_avg"] - self.base_live_wt
        self.carcass_df["excess_feed"] = self.carcass_df.apply(self.calculate_excess_feed, axis=1)
        self.carcass_df["excess_feed_total"] = self.carcass_df["excess_feed"] * self.carcass_df["num"]
        self.carcass_df["excess_feed_total_cost"] = self.carcass_df["excess_feed_total"] * self.feed_price
        
    def calculate_excess_feed(self, x):
        return self.growth_model.awfi.integrate(
                self.growth_model.awg.calc_week( self.base_live_wt ) ,
                self.growth_model.awg.calc_week( x["live_avg"] ) )

    def calculate_death(self):
        self.death_size = self.growth_model.death.integrate(
                self.growth_model.awg.calc_week( self.base_live_wt ) ,
                self.growth_model.awg.calc_week( self.carcass_avg / ( self.yield_avg / 100 ) ) )

        self.market_size_adj = self.market_size - self.death_size
        self.death_cost = self.death_size * self.avg("rev_head")

    def sum(self, ColumnStr):
        return self.carcass_df[ColumnStr].sum()

    def avg(self, ColumnStr):
        return self.carcass_df[ColumnStr].mean()

    def wt_avg(self, ColumnStr):
        return ( self.carcass_df[ColumnStr].sum() * self.carcass_df["num"].sum() ) / self.carcass_df["num"].sum()

class PigGrowthModel():
    def __init__(self, awgModel = [0], awfcModel = [0], deathModel = [0], StartWeight = 0,
                 awgAdjust = None, awfcAdjust = None, PriceCutoff = [0], WtCutoff = WT_CUTOFF_DEFAULT, WeekCutoff = None):
        
        self.start_weight = StartWeight

        self.awg = Model( polynomial(awgModel) )
        if awgAdjust is not None:
            self.shift_awg(awgAdjust[0], awgAdjust[1], awgAdjust[2])

        self.awfc = Model( polynomial(awfcModel) )
        if awfcAdjust is not None:
            self.shift_awfc(awfcAdjust[0], awfcAdjust[1], awfcAdjust[2])

        self.set_gain
        self.set_fc_cum
        self.set_awfi
        self.set_awfi_cum

        self.cutoff_price = PriceCutoff
        self.cutoff_wt = WtCutoff

        if WeekCutoff is None:
            self.calc_week_cutoff

        self.feed_cost = Model(lambda x: self.heaviside_combined(x))
        self.feed_cost_total = Model(lambda x: self.feed_cost.model(x) * self.awfi.model(x))

    def shift_awg(self, zero, lb, ub):
        self.awg = Model( self.awg.model * newton(lambda x: quad(self.awg.model * x, lb, ub)[0] - zero, 1))

    def shift_awfc(self, zero, lb, ub):
        self.awfc = Model( polynomial( [1, newton(lambda x: ( self.awfc.integrate(lb, ub, ub - lb, x) ) - zero, 0)] ) )

    @property
    def set_gain(self):
        p = self.awg.model.integ()
        self.gain = Model( polynomial(p.coef) )

    @property
    def set_fc_cum(self):
        p = self.awfc.model.integ()
        m = polynomial(p.coef)
        self.fc_cum = Model(lambda x: m(x) / x )

    @property
    def set_awfi(self):
        p = self.awg.model * self.awfc.model
        self.awfi = Model( polynomial(p.coef) )

    @property
    def set_awfi_cum(self):
        p = self.awfi.model.integ()
        self.awfi_cum = Model( polynomial(self.awfi) )    

    @property
    def calc_week_cutoff(self):
        self.cutoff_week = []
        for wt in self.cutoff_wt:
            self.cutoff_week.append(self.awg.calc_week(wt))

        self.cutoff_week.append(26)

    def heaviside(self, x):
        return 0.5 + 0.5 * numpy.tanh(5000 * x)

    def heaviside_combined(self, x):
        func = self.cutoff_price[0] * self.heaviside(x - self.cutoff_week[0])
        for i in range(1, len( self.cutoff_week ) - 1, 1):
            func = func + (self.cutoff_price[i] - self.cutoff_price[i - 1]) * self.heaviside(x - self.cutoff_week[i])  

        return func 

class BarnModel():
    def __init__(self, DeathModel, PigModel, BarnSize = 2500, DeathLossPer = 3.25):
        self.death = Model( polynomial(DeathModel) )
        self.barn_size = BarnSize
        self.death_loss = DeathLossPer

        self.adjust_death

        self.pig = PigModel

        self.awg = Model(lambda x: self.pig.awg.model(x) * self.alive_shift.model(x))
        self.awfc = Model(lambda x: self.pig.awfc.model(x) * self.alive_shift.model(x))
        self.awfi = Model(lambda x: self.pig.awfi.model(x) * self.alive_shift.model(x))
        self.feed_cost_total = Model(lambda x: self.pig.feed_cost_total.model(x) * self.alive_shift.model(x))

    @property
    def adjust_death(self):
        adjust = self.death.integrate(0, 26)
        self.death.model.coef = self.death.model.coef * (self.death_loss / 100 * self.barn_size / adjust)
        self.set_alive
        self.shift_death

    @property
    def set_alive(self):
        p = self.barn_size - self.death.model.integ()
        self.alive = Model( polynomial(p.coef) )

    @property
    def shift_death(self):
        roots = self.death.model.roots()
        self.death_shift = Model(lambda x: self.death.model(x + roots[0]))
        self.alive_shift = Model(lambda x: self.alive.model(x + roots[0]))

class Model():
    def __init__(self, func):
        self.model = func

    def integrate(self, lb = 0, ub = 26, div = 1, x = None, coef = 1): 
        if x is not None:
            self.model.coef[coef] = x

        return quad(self.model, lb, ub, limit=1000)[0] / div

    def calc_week(self, zero, lb = 0):
        return newton(lambda x: self.integrate(lb, x) - zero, 26)