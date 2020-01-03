

import pandas as pd
from sqlalchemy import *
from datetime import datetime, timedelta
import numpy as np
from config import *
from optparse import OptionParser
from ipdb import set_trace

# payback and resid is calculated in factor_return_barra.py

industries = [
    '6710100000',
    '6715100000',
    '6720100000',
    '6720200000',
    '6720300000',
    '6725100000',
    '6725200000',
    '6725300000',
    '6725400000',
    '6725500000',
    '6730100000',
    '6730200000',
    '6730300000',
    '6735100000',
    '6735200000',
    '6740100000',
    '6740200000',
    '6740400000',
    '6745100000',
    '6745200000',
    '6745300000',
    '6750200000',
    '6755100000',
    '6760100000',
    ]
 
# load factor return 
def factorReturn(sdate, edate):

    db = create_engine(uris['multi_factor'])
    meta = MetaData(bind = db)
    t = Table('factor_return_barra', meta, autoload = True)
    columns = [
        t.c.trade_date,
        t.c.country,
        t.c.volatility,
        t.c.dividend_yield,
        t.c.quality,
        t.c.momentum,
        t.c.short_term_reverse,
        t.c.value,
        t.c.linear_size,
        t.c.nonlinear_size,
        t.c.growth,
        t.c.liquidity,
        t.c.sentiment,
        t.c.industry_6710100000,
        t.c.industry_6715100000,
        t.c.industry_6720100000,
        t.c.industry_6720200000,
        t.c.industry_6720300000,
        t.c.industry_6725100000,
        t.c.industry_6725200000,
        t.c.industry_6725300000,
        t.c.industry_6725400000,
        t.c.industry_6725500000,
        t.c.industry_6730100000,
        t.c.industry_6730200000,
        t.c.industry_6730300000,
        t.c.industry_6735100000,
        t.c.industry_6735200000,
        t.c.industry_6740100000,
        t.c.industry_6740200000,
        t.c.industry_6740400000,
        t.c.industry_6745100000,
        t.c.industry_6745200000,
        t.c.industry_6745300000,
        t.c.industry_6750200000,
        t.c.industry_6755100000,
        t.c.industry_6760100000,
    ]
    sql = select(columns)
    sql = sql.where(t.c.trade_date >= sdate)
    sql = sql.where(t.c.trade_date <= edate)
    payback = pd.read_sql(sql, db)

    return payback

# load regression resid for every stock or some stocks
def regressionResid(sdate, edate, stocks = None):

    db = create_engine(uris['multi_factor'])
    meta = MetaData(bind = db)
    t = Table('regression_resid_barra', meta, autoload = True)
    columns = [
        t.c.trade_date,
        t.c.stock_id,
        t.c.resid,
    ]
    sql = select(columns)
    sql = sql.where(t.c.trade_date >= sdate)
    sql = sql.where(t.c.trade_date <= edate)
    if stocks != None:
        sql = sql.where(t.c.stock_id.in_(stocks))
    resid = pd.read_sql(sql, db)
    print(len(set(resid['stock_id'])))
    set_trace()

    return resid

# load factor exposure of every stock or some stocks
def factorExposure(date, industries, stocks = None):

    db = create_engine(uris['multi_factor'])
    meta = MetaData(bind = db)
    t = Table('factor_exposure_barra', meta, autoload = True)
    columns = [
        t.c.trade_date,
        t.c.stock_id,
        #t.c.country,
        t.c.volatility,
        t.c.dividend_yield,
        t.c.quality,
        t.c.momentum,
        t.c.short_term_reverse,
        t.c.value,
        t.c.linear_size,
        t.c.nonlinear_size,
        t.c.growth,
        t.c.liquidity,
        t.c.sentiment,
        t.c.industry, # need further treatment
        #t.c.weight, # need further treatment
    ]
    sql = select(columns)
    sql = sql.where(t.c.trade_date == date)
    exposure = pd.read_sql(sql, db)

    w = exposure
    w['country'] = 1
    
    for industry in industries:
        w['industry_'+industry] = 0
    for i in range(len(w)):
        industry = w['industry'][i]
        w['industry_'+industry][i] = 1
    
    w = w.drop('industry', axis = 1)
    print(len(set(w['stock_id'])))
    set_trace()

    return w

# calculate factor fluctuation rate
def FactorFluctuation(factorReturn):
    
    flr = factorReturn.std()

    return flr

# calculate covariance matirx
def FactorCovariance(w, sigma, omiga):

    covarianceMatrix = np.dot(np.dot(w,sigma),w.T) + omiga

    return covarianceMatrix

# main function
def handle(sdate, edate, date):
    
    fr = factorReturn(sdate, edate)
    fr.set_index('trade_date', inplace = True)
    fr.sort_index(ascending = True, inplace = True)
    fr = fr.fillna(0)
   
    flr = FactorFluctuation(fr)
    print('fluctuation rate of every factors are as folllows:')
    print(flr)
    
    resid = regressionResid(sdate, edate)
    resid.sort_values(by = ['trade_date','stock_id'],ascending = True, inplace = True)
    resid.set_index(['trade_date','stock_id'], inplace = True)
    resid = resid.unstack()['resid'].fillna(0)
   
    weight = factorExposure(date, industries)
    weight.sort_values(by = ['trade_date','stock_id'],ascending = True, inplace = True)
    weight.set_index(['trade_date','stock_id'], inplace = True)
    w = np.matrix(weight)
    print(np.shape(w))
    
    sigma = np.cov(np.matrix(fr).T)
    print(np.shape(sigma))
    
    omiga = np.diag(resid.apply(lambda x: x**2).mean())
    print(np.shape(omiga))
    
    covarianceMatrix = FactorCovariance(w, sigma, omiga)
    
    print('covarianceMatrix of '+factor+' is')
    print(covarianceMatrix)
    print('task finished!')


if __name__ == '__main__':
    opt = OptionParser()
    endDate = pd.Timestamp(datetime.today()).strftime('%Y-%m-%d')
    startDate = str(int(endDate[0:4])-1)+'-01-01'
    defaultDate = pd.Timestamp(datetime.today() - timedelta(days = 1)).strftime('%Y-%m-%d')
    defaultDate = '2019-12-31'
    opt.add_option('-s','--sdate',help = 'start date', default = startDate)
    opt.add_option('-e','--edate',help = 'end date', default = endDate)
    opt.add_option('-d','--date', help = 'date', default = defaultDate)
    opt, arg = opt.parse_args()
    handle(opt.sdate, opt.edate, opt.date)

