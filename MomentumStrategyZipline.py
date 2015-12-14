
from zipline.api import order_target, record, symbols, symbol, history, add_history
import numpy as np 
import pandas as pd
import math
import pandas_datareader.data as web
import datetime
from zipline.data import load_from_yahoo
from zipline.api import (
            order_percent,
            order_target,
            order_target_percent,
            order_target_value,
            order_value,
        )

from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)

from zipline.finance.risk.risk import (

    sharpe_ratio,

    )

from zipline.finance.risk.period import (

    RiskMetricsPeriod, 

    )


from zipline.finance import commission
from zipline.data.benchmarks import (

    get_benchmark_returns

    )

import pyfolio as pf

from zipline.utils.events import date_rules, time_rules
    
window = 10 

symbolst = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']

def initialize(context):

    add_history(190, '1d', 'price')
    add_history(365, '1d', 'price')

    context.syms = symbols('AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM')

    context.stocks_to_long = 10
    context.stocks_to_short = 10

    context.day = 0
    context.n = 0

    context.weekCounter = 0 
    


    algo.set_commission(commission.PerShare(cost=0.15))

    

def handle_data(context, data):

    # Skip first 300 days to get full windows
    context.n += 1
    if context.n < 359:
        return
    context.day += 1

    if context.day < 7:
        return 
    # Compute averages
    # history() has to be called with the same params
    # from above and returns a pandas dataframe.
    historical_data =history(365, '1d', 'price')


    pastReturns = (historical_data - historical_data.shift(-1)) / historical_data.shift(-1)

    short_mavg = history(42, '1d', 'price').mean()
    long_mavg = history(84, '1d', 'price').mean()

    diff = short_mavg / long_mavg - 1

    diff = diff.dropna()
    diff.sort()


    buys = diff [diff > 0.03]
    sells = diff[diff < -0.03]
    

    buy_length = min(context.stocks_to_long, len(buys))
    short_length = min(context.stocks_to_short, len(sells))
    buy_weight = 1.0/buy_length if buy_length != 0 else 0 
    short_weight = -1.0/short_length if short_length != 0 else 0 

    buys.sort(ascending=False)
    sells.sort()
    buys = buys.iloc[:buy_length] if buy_weight != 0 else None
    sells = sells.iloc[:short_length] if short_weight != 0 else None




    stops =  historical_data.iloc[-1] * 0.05
    
    for i in range(len(context.syms)):

        #: If the security exists in our sells.index then sell
        if sells is not None and context.syms[i] in sells.index:
            
            #print ('SHORT: %s'%context.syms[i])
           order_target_percent(context.syms[i], short_weight)
            #print 'sell'
            
        #: If the security instead, exists in our buys index, buy
        elif buys is not None and context.syms[i] in buys.index:
           # print ('BUYS: %s'%context.syms[i])
            order_target_percent(context.syms[i], buy_weight)
            #print 'nothing' 
            

        #: If the security is in neither list, exit any positions we might have in that security
        else:
            order_target(context.syms[i], 0)
    
       
    context.day = 0

    # Keep track of the number of long and short positions in the portfolio

    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        if position.amount < 0:
            shorts += 1
    

    record(short_mavg=short_mavg[context.syms[1]], long_mavg=long_mavg[context.syms[1]], portfoliovalue = (context.portfolio.returns), long_count=longs, short_count=shorts)

# Note: this function can be removed if running
# this algorithm on quantopian.com
def analyze(context=None, results=None):
    import matplotlib.pyplot as plt
    import logbook
    
    logbook.StderrHandler().push_application()
    log = logbook.Logger('Algorithm')

    '''
    for i in range(len(results.portfolio_value)):
        print results.portfolio_value[i]
        print results.portfoliovalue[i]


    '''
    
    dwRet = ((data.dowJones['Close'] - data.dowJones['Close'].shift(1)) / data.dowJones['Close'].shift(1))*100
    dwRet = np.cumsum(dwRet)
    

    trans = results.ix[[t != 100000 for t in results.portfolio_value]]
    trans = results.ix[[t != 0 for t in results.algorithm_period_return]]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    Momentum_Portfolio = trans.algorithm_period_return*100
    #SP = results.benchmark_period_return[365:]*100
    SP = dwRet

    longPositions = results.long_count
    shortPositions = results.short_count

    ax1.plot(Momentum_Portfolio, label="Momentum Portfolio")
    ax1.plot(SP, label="SPY")

    ax1.legend(loc = 2)

    ax1.set_ylabel('Cumulative Return %')
    
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('Positions')
    ax2.plot(longPositions, label="long positions")
    ax2.plot(shortPositions, label="short positions")
    ax2.legend(loc = 2)

    plt.ylim(ymax = 20, ymin = -5)
    

    print len(results.transactions)

    trans = results.ix[[t != [] for t in results.transactions]]
    buys = trans.ix[[t[0]['amount'] > 0 for t in
                         trans.transactions]]
    
    
    returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(trans)
    pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
                          gross_lev=gross_lev, live_start_date='2013-10-14')
    
    

    for i in range(len(trans.transactions)):
      for n in range(len(trans.transactions[i])):
         print symbolst[trans.transactions[i][n]['sid']],":  " , trans.transactions[i][n]
    

    plt.show()

    

    plt.show()
if __name__ == '__main__':
    from datetime import datetime
    import pytz
    from zipline.algorithm import TradingAlgorithm
    from zipline.utils.factory import load_from_yahoo

    # Set the simulation start and end dates.
    start = datetime(2005, 11, 14, 0, 0, 0, 0, pytz.utc)
    end = datetime(2013, 10, 14, 0, 0, 0, 0, pytz.utc)

    # Load price data from yahoo.

    data = load_from_yahoo(stocks=['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM'], indexes={}, start=start,
                           end=end)
    
    benchmarks = get_benchmark_returns("^DJI", start, end)
    
    data.bench = benchmarks
    end2 = datetime(2013, 10, 14, 0, 0, 0, 0, pytz.utc)
    start2 = datetime(2007, 5, 07, 0, 0, 0, 0, pytz.utc)

    dw= web.DataReader("^GSPC", "yahoo", start=start2, end = end2)
    data.dowJones = dw
    
    # Create and run the algorithm.
    algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data,
                            identifiers=['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM'])


    results = algo.run(data)


    analyze(results=results)


