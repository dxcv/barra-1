import MySQLdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import statsmodels.api as sm

# part I
data_day_descriptor = pd.read_csv(r'D:\Risk_Model\Data\data_day_descriptor_raw.csv')
data_day_descriptor['TRADE_DT'] = pd.to_datetime(data_day_descriptor['TRADE_DT'].astype(str))

data_day_price = pd.read_csv(r'D:\Risk_Model\Data\data_day_price_raw.csv')
data_day_price['TRADE_DT'] = pd.to_datetime(data_day_price['TRADE_DT'].astype(str))

data_index = pd.read_csv(r'D:\Risk_Model\Data\Index_price.csv')
data_index.rename(columns={'Unnamed: 0': 'TRADE_DT', 'CLOSE': 'INDEX_CLOSE'}, inplace=True)
data_index['TRADE_DT'] = pd.to_datetime(data_index['TRADE_DT'].astype(str))

data_day_descriptor = pd.merge(data_day_descriptor, data_day_price, how='right', on=['S_INFO_WINDCODE', 'TRADE_DT'], sort=False)
data_day_descriptor = pd.merge(data_day_descriptor, data_index, how='left', on='TRADE_DT', sort=False)
data_day_descriptor['S_I_CLOSE'] = data_day_descriptor['S_DQ_ADJCLOSE']*100 + data_day_descriptor['INDEX_CLOSE']/10000
retained_columns = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_VAL_MV', 'S_DQ_MV', 'S_VAL_PB_NEW', 'S_VAL_PE_TTM', 'S_DQ_FREETURNOVER',
                    'S_PRICE_DIV_DPS', 'S_DQ_CLOSE_TODAY', 'S_DQ_ADJCLOSE', 'S_PQ_ADJHIGH_52W', 'S_PQ_ADJLOW_52W', 'S_I_CLOSE']
loc_t = (data_day_descriptor.TRADE_DT >= datetime(2006, 1, 1)) & (data_day_descriptor.TRADE_STATUS == 1)
data_day_descriptor = data_day_descriptor.loc[loc_t, retained_columns]
# BTOP ETOP DTOP
data_day_descriptor[['S_VAL_PB_NEW', 'S_VAL_PE_TTM', 'S_PRICE_DIV_DPS']] = 1 / data_day_descriptor[['S_VAL_PB_NEW', 'S_VAL_PE_TTM', 'S_PRICE_DIV_DPS']]
data_day_descriptor.rename(columns={'S_INFO_WINDCODE': 'WIND_CODE', 'S_VAL_PB_NEW': 'BTOP', 'S_VAL_PE_TTM': 'ETOP', 'S_PRICE_DIV_DPS': 'DTOP'}, inplace=True)

columns_t = ['BETA', 'RS_M', 'RS_S', 'STREV', 'RSTR', 'ALPHA', 'HALPHA', 'DASTD', 'CMRA', 'HSIGMA', 'STOM', 'STOQ', 'STOA', 'TODAY_RETURN', 'NEXT_RETURN']
for i in columns_t:
    data_day_descriptor[i] = np.nan
data_day_descriptor['CMRA'] = np.log(data_day_descriptor['S_PQ_ADJHIGH_52W']) - np.log(data_day_descriptor['S_PQ_ADJLOW_52W'])


def yn_liquidity(data):
    fun_data = data[~np.isnan(data)]
    if fun_data.shape[0] == 0:
        return np.nan
    else:
        fun_sum = np.sum(fun_data) * 21 / fun_data.shape[0]
        return np.log(fun_sum)


def yn_rs(data, half_life):
    fun_data = data[~np.isnan(data)]
    if fun_data.shape[0] < half_life:
        return np.nan
    else:
        stock_price = np.floor(fun_data)
        index_price = (fun_data - stock_price) * 10000
        stock_price = stock_price / 100
        stock_return = np.log(stock_price[1:]) - np.log(stock_price[:-1])
        index_return = np.log(index_price[1:]) - np.log(index_price[:-1])
        relative_return = stock_return - index_return
        fun_lambda = 0.5 ** (1 / half_life)
        fun_weights = fun_lambda ** (np.arange(len(relative_return) - 1, -1, -1))
        relative_strength = np.dot(relative_return, fun_weights)
        return relative_strength


def yn_wls(data, half_life, params=1):
    fun_data = data[~np.isnan(data)]
    if fun_data.shape[0] < half_life:
        return np.nan
    else:
        stock_price = np.floor(fun_data)
        index_price = (fun_data - stock_price) * 10000
        stock_price = stock_price / 100
        y = np.log(stock_price[1:]) - np.log(stock_price[:-1])
        x = np.log(index_price[1:]) - np.log(index_price[:-1])
        fun_lambda = 0.5 ** (1 / half_life)
        fun_weights = fun_lambda ** (np.arange(len(y) - 1, -1, -1))
        wls_results = sm.WLS(y, sm.add_constant(x), weights=fun_weights ** 2).fit()
        if params == 0:
            return wls_results.params[0]
        elif params == 1:
            return wls_results.params[1]
        elif params == 2:
            fun_var = np.cov(wls_results.resid, ddof=0, aweights=fun_weights)
            return np.sqrt(np.float(fun_var))
        else:
            return None


def yn_ewm_std(data, half_life):
    fun_data = data[~np.isnan(data)]
    if fun_data.shape[0] < half_life:
        return np.nan
    else:
        fun_lambda = 0.5 ** (1 / half_life)
        fun_weights = fun_lambda ** (np.arange(len(fun_data) - 1, -1, -1))
        fun_var = np.cov(fun_data, ddof=0, aweights=fun_weights)
        return np.sqrt(np.float(fun_var))


data_day_descriptor.drop_duplicates(subset=['WIND_CODE', 'TRADE_DT'], inplace=True)
data_day_descriptor.set_index(['WIND_CODE'], inplace=True)
data_day_descriptor.sort_values(by=['WIND_CODE', 'TRADE_DT'], inplace=True)
processed_len = len(data_day_descriptor.index.get_level_values(0).unique())
for i_num, i_code in enumerate(data_day_descriptor.index.get_level_values(0).unique()):
    if data_day_descriptor.loc[[i_code]].shape[0] < 2:
        continue
    # Liquidity
    data_day_descriptor.loc[i_code, 'STOM'] = data_day_descriptor.loc[i_code, 'S_DQ_FREETURNOVER'].rolling(21).apply(lambda x: yn_liquidity(data=x), 'raw=True')
    data_day_descriptor.loc[i_code, 'STOQ'] = data_day_descriptor.loc[i_code, 'S_DQ_FREETURNOVER'].rolling(63).apply(lambda x: yn_liquidity(data=x), 'raw=True')
    data_day_descriptor.loc[i_code, 'STOA'] = data_day_descriptor.loc[i_code, 'S_DQ_FREETURNOVER'].rolling(252).apply(lambda x: yn_liquidity(data=x), 'raw=True')
    # Momentum
    data_day_descriptor.loc[i_code, 'RS_S'] = data_day_descriptor.loc[i_code, 'S_I_CLOSE'].rolling(63+1).apply(lambda x: yn_rs(x, half_life=10), 'raw=True')
    data_day_descriptor.loc[i_code, 'STREV'] = data_day_descriptor.loc[i_code, 'RS_S'].rolling(3).apply(lambda x: np.mean(x), 'raw=True')
    data_day_descriptor.loc[i_code, 'RS_M'] = data_day_descriptor.loc[i_code, 'S_I_CLOSE'].rolling(252+1).apply(lambda x: yn_rs(x, half_life=126), 'raw=True')
    data_day_descriptor.loc[i_code, 'RSTR'] = data_day_descriptor.loc[i_code, 'RS_M'].rolling(21).apply(lambda x: np.mean(x[:11]), 'raw=True')
    data_day_descriptor.loc[i_code, 'ALPHA'] = data_day_descriptor.loc[i_code, 'S_I_CLOSE'].rolling(252+1).apply(lambda x: yn_wls(x, half_life=63, params=0), 'raw=True')
    data_day_descriptor.loc[i_code, 'HALPHA'] = data_day_descriptor.loc[i_code, 'ALPHA'].rolling(21).apply(lambda x: np.mean(x[:11]), 'raw=True')
    # Volatility
    data_day_descriptor.loc[i_code, 'BETA'] = data_day_descriptor.loc[i_code, 'S_I_CLOSE'].rolling(252+1).apply(lambda x: yn_wls(data=x, half_life=63, params=1), 'raw=True')
    data_day_descriptor.loc[i_code, 'TODAY_RETURN'] = data_day_descriptor.loc[i_code, 'S_DQ_ADJCLOSE'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), 'raw=True')
    data_day_descriptor.loc[i_code, 'DASTD'] = data_day_descriptor.loc[i_code, 'TODAY_RETURN'].rolling(252).apply(lambda x: yn_ewm_std(data=x, half_life=42), 'raw=True')
    data_day_descriptor.loc[i_code, 'HSIGMA'] = data_day_descriptor.loc[i_code, 'S_I_CLOSE'].rolling(252+1).apply(lambda x: yn_wls(data=x, half_life=63, params=2), 'raw=True')
    # next return
    t_next_return = data_day_descriptor.loc[i_code, 'TODAY_RETURN'].values  # TODAY_RETURN中的值也会改变
    t_next_return[:-1] = t_next_return[1:]
    t_next_return[-1] = np.nan
    data_day_descriptor.loc[i_code, 'NEXT_RETURN'] = t_next_return
    if i_num % 100 == 0:
        print('Part_I已完成：', int(i_num * 100 / processed_len), '%')

retained_columns = ['TRADE_DT', 'BTOP', 'ETOP', 'S_VAL_MV', 'S_DQ_MV', 'STREV', 'RSTR', 'HALPHA', 'DTOP', 'BETA', 'DASTD',
                    'CMRA', 'HSIGMA', 'STOM', 'STOQ', 'STOA', 'NEXT_RETURN', 'S_DQ_CLOSE_TODAY']
loc_t = data_day_descriptor.TRADE_DT >= datetime(2008, 1, 1)
data_day_descriptor = data_day_descriptor.loc[loc_t, retained_columns]

data_day_descriptor.to_csv(r'D:\Risk_Model\Data\data_day_descriptor_p1.csv')


# Part II: Financial descriptor
conn = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='123456', db='test', charset='utf8')

columns_BS = 'WIND_CODE,ACTUAL_ANN_DT,REPORT_PERIOD,STATEMENT_TYPE,LT_BORROW,BONDS_PAYABLE,LT_PAYABLE,SPECIFIC_ITEM_PAYABLE,TOT_SHRHLDR_EQY_EXCL_MIN_INT,TOT_CUR_LIAB,' \
             'TOT_CUR_ASSETS,MONETARY_CAP,TRADABLE_FIN_ASSETS,ST_BORROW,TRADABLE_FIN_LIAB,NON_CUR_LIAB_DUE_WITHIN_1Y,TOT_ASSETS,MINORITY_INT,TOT_LIAB,CAP_STK'
sql = 'select '+columns_BS+' from asharebalancesheet'
data_BS = pd.read_sql(sql, conn)

columns_IS = 'WIND_CODE,REPORT_PERIOD,STATEMENT_TYPE,NET_PROFIT_AFTER_DED_NR_LP,NET_PROFIT_EXCL_MIN_INT_INC,OPER_REV,LESS_OPER_COST,TOT_PROFIT,LESS_FIN_EXP'  # notice
sql = 'select '+columns_IS+' from ashareincome'
data_IS = pd.read_sql(sql, conn)

columns_CF = 'WIND_CODE,REPORT_PERIOD,STATEMENT_TYPE,DEPR_FA_COGA_DPBA,AMORT_INTANG_ASSETS,AMORT_LT_DEFERRED_EXP,' \
             'NET_CASH_FLOWS_OPER_ACT,DECR_INVENTORIES,DECR_OPER_PAYABLE,INCR_OPER_PAYABLE,OTHERS,STOT_CASH_OUTFLOWS_INV_ACT'
sql = 'select '+columns_CF+' from asharecashflow'
data_CF = pd.read_sql(sql, conn)

# B/S
data_BS.REPORT_PERIOD = pd.to_datetime(data_BS.REPORT_PERIOD)
data_BS['month'] = data_BS.REPORT_PERIOD.map(lambda x: x.month in [3, 6, 9, 12])
loc_t = (data_BS.STATEMENT_TYPE == '408001000') & data_BS.month
data_BS = data_BS.loc[loc_t]
data_BS.drop_duplicates(subset=['WIND_CODE', 'REPORT_PERIOD'], inplace=True)
columns_t = ['LT_BORROW', 'BONDS_PAYABLE', 'LT_PAYABLE', 'SPECIFIC_ITEM_PAYABLE']
data_BS['LD'] = data_BS[columns_t].sum(axis=1, skipna=True)
data_BS['BV_EM'] = data_BS['TOT_SHRHLDR_EQY_EXCL_MIN_INT']
data_BS['CL'] = data_BS['TOT_CUR_LIAB']
columns_t_add = ['TOT_CUR_ASSETS', 'ST_BORROW', 'TRADABLE_FIN_LIAB', 'NON_CUR_LIAB_DUE_WITHIN_1Y']
columns_t_sub = ['MONETARY_CAP', 'TRADABLE_FIN_ASSETS', 'TOT_CUR_LIAB']
data_BS['WC_BS'] = data_BS[columns_t_add].sum(axis=1, skipna=True) - data_BS[columns_t_sub].sum(axis=1, skipna=True)
data_BS['TA'] = data_BS['TOT_ASSETS']
data_BS['EV_EXCL_MV'] = data_BS[['MINORITY_INT', 'TOT_LIAB']].sum(axis=1, skipna=True) - data_BS[['MONETARY_CAP', 'TRADABLE_FIN_ASSETS']].sum(axis=1, skipna=True)
data_BS['SHARES'] = data_BS['CAP_STK']
# CAP_STK 股本
# I/S
loc_t = data_IS.STATEMENT_TYPE == '408001000'
data_IS_A = data_IS.loc[loc_t]
loc_t = data_IS.STATEMENT_TYPE == '408002000'
data_IS_Q = data_IS.loc[loc_t]
data_IS = pd.merge(data_IS_A, data_IS_Q[['WIND_CODE', 'REPORT_PERIOD', 'NET_PROFIT_AFTER_DED_NR_LP', 'OPER_REV']], how='left', on=['WIND_CODE', 'REPORT_PERIOD'], suffixes=('_A', '_Q'), sort=False)
data_IS['EBIT'] = data_IS['TOT_PROFIT'] + data_IS['LESS_FIN_EXP'].fillna(0)
data_IS['EARNINGS_A'] = data_IS['NET_PROFIT_AFTER_DED_NR_LP_A']
data_IS['EARNINGS_Q'] = data_IS['NET_PROFIT_AFTER_DED_NR_LP_Q']  # 季报无此指标
data_IS['EARNINGS_2'] = data_IS['NET_PROFIT_EXCL_MIN_INT_INC']  # use to calculate CETOE
data_IS['SALES_A'] = data_IS['OPER_REV_A']
data_IS['SALES_Q'] = data_IS['OPER_REV_Q']
data_IS['COGS'] = data_IS['LESS_OPER_COST']
# NET_PROFIT_AFTER_DED_NR_LP_Q全部为null
data_IS.REPORT_PERIOD = pd.to_datetime(data_IS.REPORT_PERIOD)
data_IS['month'] = data_IS.REPORT_PERIOD.map(lambda x: x.month in [3, 6, 9, 12])
data_IS = data_IS.loc[data_IS.month]
data_IS.drop_duplicates(subset=['WIND_CODE', 'REPORT_PERIOD'], inplace=True)
data_IS.set_index('REPORT_PERIOD', inplace=True)
data_IS.sort_values(by=['REPORT_PERIOD', 'WIND_CODE'], inplace=True)
for i_num, i_date in enumerate(data_IS.index.unique()):
    if i_date.year < 2003:
        continue

    if i_date.month == 3:
        data_IS.loc[i_date, 'EARNINGS_Q'] = data_IS.loc[i_date, 'EARNINGS_A']
    else:
        month_t = i_date.month - 3
        if month_t == 9:
            day_t = 30
        elif month_t == 6:
            day_t = 30
        else:
            day_t = 31
        i_date_adv_3m = datetime(i_date.year, month_t, day_t)
        columns_t = ['WIND_CODE', 'EARNINGS_A']
        data_IS_t = pd.merge(data_IS.loc[i_date, columns_t], data_IS.loc[i_date_adv_3m, columns_t], how='left', on='WIND_CODE', sort=False, suffixes=('_now', '_adv_3m'))
        data_IS_t['EARNINGS_Q'] = data_IS_t['EARNINGS_A_now'] - data_IS_t['EARNINGS_A_adv_3m']
        data_IS.loc[i_date, 'EARNINGS_Q'] = data_IS_t['EARNINGS_Q'].values
data_IS.reset_index(inplace=True)
# CF
data_CF.REPORT_PERIOD = pd.to_datetime(data_CF.REPORT_PERIOD)
data_CF['month'] = data_CF.REPORT_PERIOD.map(lambda x: x.month in [3, 6, 9, 12])
loc_t = (data_CF.STATEMENT_TYPE == '408001000') & data_CF.month
data_CF = data_CF.loc[loc_t]
data_CF.drop_duplicates(subset=['WIND_CODE', 'REPORT_PERIOD'], inplace=True)
data_CF['WC_CF'] = data_CF[['DECR_INVENTORIES', 'DECR_OPER_PAYABLE', 'INCR_OPER_PAYABLE', 'OTHERS']].sum(axis=1, skipna=True, min_count=1)
data_CF['D_A'] = data_CF[['DEPR_FA_COGA_DPBA', 'AMORT_INTANG_ASSETS', 'AMORT_LT_DEFERRED_EXP']].sum(axis=1, skipna=True, min_count=1)
data_CF['CFO'] = data_CF['NET_CASH_FLOWS_OPER_ACT']
data_CF['CAP_EXP'] = data_CF['STOT_CASH_OUTFLOWS_INV_ACT']
# notice: 所有利用fiscal year计算的指标都最后再加入

columns_BS = ['WIND_CODE', 'ACTUAL_ANN_DT', 'REPORT_PERIOD', 'LD', 'BV_EM', 'CL', 'WC_BS', 'TA', 'EV_EXCL_MV', 'SHARES']
columns_IS = ['WIND_CODE', 'REPORT_PERIOD', 'EBIT', 'EARNINGS_A', 'EARNINGS_Q', 'EARNINGS_2', 'SALES_A', 'SALES_Q', 'COGS']
columns_CF = ['WIND_CODE', 'REPORT_PERIOD', 'WC_CF', 'D_A', 'CFO', 'CAP_EXP']

data_FS = pd.merge(data_BS[columns_BS], data_IS[columns_IS], how='left', on=['WIND_CODE', 'REPORT_PERIOD'], sort=False)
data_FS = pd.merge(data_FS, data_CF[columns_CF], how='left', on=['WIND_CODE', 'REPORT_PERIOD'], sort=False)
########################################################################################################################
data_FS.drop_duplicates(subset=['REPORT_PERIOD', 'WIND_CODE'], inplace=True)
data_FS.set_index('REPORT_PERIOD', inplace=True)
data_FS.sort_values(by=['REPORT_PERIOD', 'WIND_CODE'], inplace=True)

data_FS['BLEV'] = (data_FS['BV_EM'] + data_FS['LD']) / data_FS['BV_EM']
data_FS['DTOA'] = (data_FS['LD'] + data_FS['CL']) / data_FS['TA']
columns_t = ['EGRO_ST', 'SGRO_ST', 'CETOE2', 'EBITDA_TTM', 'ACBS', 'ACCF', 'CETOE', 'ATO', 'GP', 'GPM', 'ROA']
for i in columns_t:
    data_FS[i] = np.nan

for i_num, i_date in enumerate(data_FS.index.unique()):
    if i_date.year < 2003:
        continue
    i_date_adv_y = datetime(i_date.year - 1, i_date.month, i_date.day)
    # 计算同比和TTM的指标
    columns_t = ['WIND_CODE', 'WC_BS', 'EARNINGS_Q', 'SALES_Q', 'TA', 'EARNINGS_A', 'EARNINGS_2', 'SALES_A', 'COGS', 'EBIT', 'WC_CF', 'D_A', 'CFO', 'CAP_EXP']
    temp_t1 = pd.merge(data_FS.loc[i_date, columns_t], data_FS.loc[i_date_adv_y, columns_t], how='left', on='WIND_CODE', suffixes=('_now', '_adv_y'), sort=False)

    i_date_LFY = datetime(i_date.year - 1, 12, 31)
    columns_t = ['WIND_CODE', 'EARNINGS_A', 'EARNINGS_2', 'SALES_A', 'COGS', 'EBIT', 'WC_CF', 'D_A', 'CFO', 'CAP_EXP']
    temp_t2 = data_FS.loc[i_date_LFY, columns_t].copy()
    temp_t2.rename(columns={'EARNINGS_A': 'EARNINGS_A_LFY', 'EARNINGS_2': 'EARNINGS_2_LFY', 'SALES_A': 'SALES_A_LFY', 'COGS': 'COGS_LFY',
                            'EBIT': 'EBIT_LFY', 'WC_CF': 'WC_CF_LFY', 'D_A': 'D_A_LFY', 'CFO': 'CFO_LFY', 'CAP_EXP': 'CAP_EXP_LFY'}, inplace=True)
    data_FS_t = pd.merge(temp_t1, temp_t2, how='left', on='WIND_CODE', suffixes=('_now', '_adv_y'), sort=False)
    # Grwoth descriptors YOY
    loc_t = data_FS_t['EARNINGS_Q_adv_y'] > 0
    data_FS_t.loc[loc_t, 'EGRO_ST'] = (data_FS_t.loc[loc_t, 'EARNINGS_Q_now'] - data_FS_t.loc[loc_t, 'EARNINGS_Q_adv_y']) / data_FS_t.loc[loc_t, 'EARNINGS_Q_adv_y']
    loc_t = data_FS_t['EARNINGS_Q_adv_y'] < 0
    data_FS_t.loc[loc_t, 'EGRO_ST'] = -(data_FS_t.loc[loc_t, 'EARNINGS_Q_now'] - data_FS_t.loc[loc_t, 'EARNINGS_Q_adv_y']) / data_FS_t.loc[loc_t, 'EARNINGS_Q_adv_y']
    loc_t = data_FS_t['EARNINGS_Q_adv_y'] == 0
    data_FS_t.loc[loc_t, 'EGRO_ST'] = np.nan
    #
    loc_t = data_FS_t['SALES_Q_adv_y'] > 0
    data_FS_t.loc[loc_t, 'SGRO_ST'] = (data_FS_t.loc[loc_t, 'SALES_Q_now'] - data_FS_t.loc[loc_t, 'SALES_Q_adv_y']) / data_FS_t.loc[loc_t, 'SALES_Q_adv_y']
    loc_t = data_FS_t['SALES_Q_adv_y'] < 0
    data_FS_t.loc[loc_t, 'SGRO_ST'] = -(data_FS_t.loc[loc_t, 'SALES_Q_now'] - data_FS_t.loc[loc_t, 'SALES_Q_adv_y']) / data_FS_t.loc[loc_t, 'SALES_Q_adv_y']
    loc_t = data_FS_t['SALES_Q_adv_y'] == 0
    data_FS_t.loc[loc_t, 'SGRO_ST'] = np.nan
    # Value descriptors TTM
    data_FS_t['E2_TTM'] = data_FS_t['EARNINGS_2_now'] + data_FS_t['EARNINGS_2_LFY'] - data_FS_t['EARNINGS_2_adv_y']
    freigth_t = data_FS_t['D_A_now'] - data_FS_t['D_A_adv_y']
    data_FS_t['D_A_TTM'] = data_FS_t['D_A_LFY'].sub(freigth_t, fill_value=0)
    data_FS_t['CETOE2'] = (data_FS_t['E2_TTM'] + data_FS_t['D_A_TTM']) / data_FS_t['E2_TTM']
    data_FS_t['EBITDA_TTM'] = data_FS_t['EBIT_now'] + data_FS_t['EBIT_LFY'] - data_FS_t['EBIT_adv_y']
    # Earnings Quality descriptors TTM
    data_FS_t['DELTA_WC_BS'] = data_FS_t['WC_BS_now'] - data_FS_t['WC_BS_adv_y']
    data_FS_t['ACBS'] = (data_FS_t['DELTA_WC_BS'] + data_FS_t['D_A_TTM']) / data_FS_t['TA_now']
    freigth_t = data_FS_t['WC_CF_now'] - data_FS_t['WC_CF_adv_y']
    data_FS_t['DELTA_WC_CF'] = data_FS_t['WC_CF_LFY'].add(freigth_t, fill_value=0)
    data_FS_t['ACCF'] = (data_FS_t['DELTA_WC_CF'] - data_FS_t['D_A_TTM']) / data_FS_t['TA_now']
    data_FS_t['E_TTM'] = data_FS_t['EARNINGS_A_now'] + data_FS_t['EARNINGS_A_LFY'] - data_FS_t['EARNINGS_A_adv_y']
    data_FS_t['CETOE'] = (data_FS_t['E_TTM'] + data_FS_t['D_A_TTM']) / data_FS_t['E_TTM']
    # Profitability desciptors TTM
    data_FS_t['SALES_TTM'] = data_FS_t['SALES_A_now'] + data_FS_t['SALES_A_LFY'] - data_FS_t['SALES_A_adv_y']
    data_FS_t['TA_AVERAGE'] = (data_FS_t['TA_now'] + data_FS_t['TA_adv_y']) / 2
    data_FS_t['ATO'] = data_FS_t['SALES_TTM'] / data_FS_t['TA_AVERAGE']
    data_FS_t['COGS_TTM'] = data_FS_t['COGS_now'] + data_FS_t['COGS_LFY'] - data_FS_t['COGS_adv_y']
    data_FS_t['GP'] = (data_FS_t['SALES_TTM'] - data_FS_t['COGS_TTM']) / data_FS_t['TA_AVERAGE']
    data_FS_t['GPM'] = (data_FS_t['SALES_TTM'] - data_FS_t['COGS_TTM']) / data_FS_t['SALES_TTM']
    data_FS_t['CFO_TTM'] = data_FS_t['CFO_now'] + data_FS_t['CFO_LFY'] - data_FS_t['CFO_adv_y']
    data_FS_t['ROA'] = data_FS_t['CFO_TTM'] / data_FS_t['TA_AVERAGE']
    columns_t = ['EGRO_ST', 'SGRO_ST', 'CETOE2', 'EBITDA_TTM', 'ACBS', 'ACCF', 'CETOE', 'ATO', 'GP', 'GPM', 'ROA']
    data_FS.loc[i_date, columns_t] = data_FS_t[columns_t].values

########################################################################################################################
data_FS.reset_index(inplace=True)
columns_t = ['WIND_CODE', 'REPORT_PERIOD', 'SALES_A', 'EARNINGS_A', 'CFO', 'SHARES', 'TA', 'CAP_EXP']
loc_t = data_FS.REPORT_PERIOD.astype(str).str.contains('12-31')
data_FS_FY = data_FS.loc[loc_t, columns_t].copy()
data_FS_FY.set_index('WIND_CODE', inplace=True)
data_FS_FY.sort_values(by=['WIND_CODE', 'REPORT_PERIOD'], inplace=True)
columns_t = ['VSAL', 'VERN', 'VFLO', 'AGRO', 'IGRO', 'CXGRO', 'EGRO_LT', 'SGRO_LT']
for i in columns_t:
    data_FS_FY[i] = np.nan


def yn_e_var(data):
    data = data[~np.isnan(data)]
    growth_list = list()
    for i in range(1, data.shape[0]):
        if data[i-1] > 0:
            growth_t = (data[i] - data[i-1]) / data[i-1]
        elif data[i-1] < 0:
            growth_t = -(data[i] - data[i-1]) / data[i-1]
        else:
            growth_t = np.nan
        growth_list.append(growth_t)
    growth_array = np.array(growth_list)
    fun_var = np.cov(growth_array)
    fun_std = np.sqrt(fun_var)
    return fun_std


def yn_growth_lt(data):
    data = data[~np.isnan(data)]
    x = np.arange(data.shape[0]) + 1
    y = data
    ols_results = sm.OLS(y, sm.add_constant(x)).fit()
    growth_t = ols_results.params[1] / np.mean(data)
    return growth_t


for i_num, i_code in enumerate(data_FS_FY.index.unique()):
    if data_FS_FY.loc[[i_code]].shape[0] < 2:
        continue
    data_FS_FY_t = data_FS_FY.loc[i_code].copy()
    data_FS_FY_t['VSAL'] = data_FS_FY_t['SALES_A'].rolling(5).apply(lambda x: yn_e_var(x), 'raw=True')
    data_FS_FY_t['VERN'] = data_FS_FY_t['EARNINGS_A'].rolling(5).apply(lambda x: yn_e_var(x), 'raw=True')
    data_FS_FY_t['VFLO'] = data_FS_FY_t['CFO'].rolling(5).apply(lambda x: yn_e_var(x), 'raw=True')
    data_FS_FY_t['AGRO'] = data_FS_FY_t['TA'].rolling(5).apply(lambda x: yn_growth_lt(x), 'raw=True')
    data_FS_FY_t['IGRO'] = data_FS_FY_t['SHARES'].rolling(5).apply(lambda x: yn_growth_lt(x), 'raw=True')
    data_FS_FY_t['CXGRO'] = data_FS_FY_t['CAP_EXP'].rolling(5).apply(lambda x: yn_growth_lt(x), 'raw=True')
    data_FS_FY_t['EGRO_LT'] = data_FS_FY_t['EARNINGS_A'].rolling(5).apply(lambda x: yn_growth_lt(x), 'raw=True')
    data_FS_FY_t['SGRO_LT'] = data_FS_FY_t['SALES_A'].rolling(5).apply(lambda x: yn_growth_lt(x), 'raw=True')
    columns_t = ['VSAL', 'VERN', 'VFLO', 'AGRO', 'IGRO', 'CXGRO', 'EGRO_LT', 'SGRO_LT']
    data_FS_FY.loc[i_code, columns_t] = data_FS_FY_t[columns_t].values
########################################################################################################################
data_FS_FY.reset_index(inplace=True)
data_FS = pd.merge(data_FS, data_FS_FY, how='left', on=['REPORT_PERIOD', 'WIND_CODE'], sort=False)
# 一季报披露时间修改
loc_t = data_FS.REPORT_PERIOD.astype(str).str.contains('12-31')
data_FS_AdjDate = data_FS.loc[loc_t, ['REPORT_PERIOD', 'WIND_CODE', 'ACTUAL_ANN_DT']].copy()
data_FS_AdjDate['REPORT_PERIOD'] = data_FS_AdjDate['REPORT_PERIOD'].map(lambda x: datetime(x.year+1, 3, 31))
data_FS = pd.merge(data_FS, data_FS_AdjDate, how='left', on=['REPORT_PERIOD', 'WIND_CODE'], sort=False)
data_FS.ACTUAL_ANN_DT_x = pd.to_datetime(data_FS.ACTUAL_ANN_DT_x)
data_FS.ACTUAL_ANN_DT_y = pd.to_datetime(data_FS.ACTUAL_ANN_DT_y)
data_FS['ACTUAL_ANN_DT'] = data_FS[['ACTUAL_ANN_DT_x', 'ACTUAL_ANN_DT_y']].max(axis=1, skipna=True)
#
retained_columns = ['REPORT_PERIOD', 'WIND_CODE', 'ACTUAL_ANN_DT', 'CETOE2', 'EV_EXCL_MV', 'EBITDA_TTM', 'LD', 'BLEV', 'DTOA', 'VSAL', 'VERN', 'VFLO',
                    'ACBS', 'ACCF', 'CETOE', 'ATO', 'GP', 'GPM', 'ROA', 'IGRO', 'AGRO', 'CXGRO', 'EGRO_ST', 'SGRO_ST', 'EGRO_LT', 'SGRO_LT']
loc_t = data_FS.WIND_CODE.str.contains('A') | data_FS.WIND_CODE.str.contains('T')
data_FS = data_FS.loc[~loc_t, retained_columns]
data_FS = data_FS.loc[data_FS.REPORT_PERIOD > datetime(2006, 1, 1)]
data_FS.drop_duplicates(subset=['REPORT_PERIOD', 'WIND_CODE'], inplace=True)

data_FS.to_csv(r'D:\Risk_Model\Data\data_FS.csv')

# Part III Forecast & Industries & Descriptors
data_day_descriptor = pd.read_csv(r'D:\Risk_Model\Data\data_day_descriptor_p1.csv')
data_day_descriptor.TRADE_DT = pd.to_datetime(data_day_descriptor.TRADE_DT)
data_FS = pd.read_csv(r'D:\Risk_Model\Data\data_FS.csv')
data_FS.rename(columns={'ACTUAL_ANN_DT': 'TRADE_DT'}, inplace=True)
data_FS.TRADE_DT = pd.to_datetime(data_FS.TRADE_DT)
data_FS.TRADE_DT = data_FS.TRADE_DT + timedelta(1)  # 延后一天
data_Forecast = pd.read_csv(r'D:\Risk_Model\Data\Forecast_descriptor.csv')
data_Forecast.TRADE_DT = pd.to_datetime(data_Forecast.TRADE_DT)
data_Industry = pd.read_csv(r'D:\Risk_Model\Data\Industry.csv')
data_Industry.TRADE_DT = pd.to_datetime(data_Industry.TRADE_DT)
data_Industry.drop_duplicates(subset=['TRADE_DT', 'WIND_CODE'], inplace=True)

# data_FS 数据的处理
data_FS.set_index('WIND_CODE', inplace=True)
data_FS.sort_values(by=['WIND_CODE', 'REPORT_PERIOD'], inplace=True)
columns_t = ['VSAL', 'VERN', 'VFLO', 'AGRO', 'IGRO', 'CXGRO', 'EGRO_LT', 'SGRO_LT']
for i_num, i_code in enumerate(data_FS.index.unique()):
    data_FS.loc[i_code, columns_t] = data_FS.loc[i_code, columns_t].fillna(method='ffill', limit=12)
loc_t = data_FS.REPORT_PERIOD.astype(str).str.contains('12-31')
data_FS = data_FS.loc[~loc_t]
data_FS.reset_index(inplace=True)
# Forecast数据的处理
data_Forecast[['WEST_NETPROFIT_FY1_3M', 'WEST_SALES_FY1_3M']] = data_Forecast[['WEST_NETPROFIT_FY1_3M', 'WEST_SALES_FY1_3M']].fillna(0)
# 合并
data_day_descriptor = pd.merge(data_day_descriptor, data_FS, how='outer', on=['WIND_CODE', 'TRADE_DT'], sort=False)
data_day_descriptor = pd.merge(data_day_descriptor, data_Forecast, how='outer', on=['WIND_CODE', 'TRADE_DT'], sort=False)
data_day_descriptor.set_index(['WIND_CODE'], inplace=True)
data_day_descriptor.sort_values(by=['WIND_CODE', 'TRADE_DT'], inplace=True)
# ffill
nan2ffill_FS = ['CETOE2', 'EV_EXCL_MV', 'EBITDA_TTM', 'LD', 'BLEV', 'DTOA', 'VSAL', 'VERN', 'VFLO', 'ACBS', 'ACCF',
                'CETOE', 'ATO', 'GP', 'GPM', 'ROA', 'IGRO', 'AGRO', 'CXGRO', 'EGRO_ST', 'SGRO_ST', 'EGRO_LT', 'SGRO_LT']
nan2ffill_Forecast = ['WEST_AVGDPS_FY1', 'WEST_EPS_FY1', 'WEST_NETPROFIT_CAGR', 'WEST_NETPROFIT_FY1_3M', 'WEST_SALES_FY1_3M', 'WEST_STDEPS_FY1']
processed_len = len(data_day_descriptor.index.unique())
for i_num, i_code in enumerate(data_day_descriptor.index.unique()):
    data_day_descriptor.loc[i_code, nan2ffill_FS] = data_day_descriptor.loc[i_code, nan2ffill_FS].fillna(method='ffill', limit=360)
    data_day_descriptor.loc[i_code, nan2ffill_Forecast] = data_day_descriptor.loc[i_code, nan2ffill_Forecast].fillna(method='ffill', limit=30)
    if i_num % 100 == 0:
        print('Part_III已完成：', int(i_num*100/processed_len), '%')

# Industry #############################################################################################################
data_day_descriptor.reset_index(inplace=True)
data_day_descriptor['TRADE_YEAR'] = data_day_descriptor['TRADE_DT'].map(lambda x: x.year)
data_Industry['TRADE_YEAR'] = data_Industry['TRADE_DT'].map(lambda x: x.year)
data_day_descriptor = pd.merge(data_day_descriptor, data_Industry[['TRADE_YEAR', 'WIND_CODE', 'INDUSTRY_CITIC']], how='left', on=['TRADE_YEAR', 'WIND_CODE'], sort=False)

# Descriptors ##########################################################################################################
# Value: BTOP; ETOP EPIBS CETOP ENMU
data_day_descriptor['EPIBS'] = data_day_descriptor['WEST_EPS_FY1'] / data_day_descriptor['S_DQ_CLOSE_TODAY']
data_day_descriptor['CETOP'] = data_day_descriptor['CETOE2'] * data_day_descriptor['ETOP']
data_day_descriptor['ENMU'] = data_day_descriptor['EBITDA_TTM'] / (data_day_descriptor['EV_EXCL_MV'] + data_day_descriptor['S_VAL_MV'])
# Size: LSIZE; MID_CAP
data_day_descriptor['LSIZE'] = np.log(data_day_descriptor['S_VAL_MV'] * 10000)
data_day_descriptor['MID_CAP'] = np.nan
# Momentum: STREV; RSTR HALPHA
# Quality: MLEV BLEV DTOA; VSAL VERN VFLO SPIBS; ACBS ACCF CETOE; ATO GP GPM ROA; IGRO AGRO CXGRO
data_day_descriptor['MLEV'] = ((data_day_descriptor['S_VAL_MV'] * 10000) + data_day_descriptor['LD']) / (data_day_descriptor['S_VAL_MV'] * 10000)
data_day_descriptor['SPIBS'] = data_day_descriptor['WEST_STDEPS_FY1'] / data_day_descriptor['S_DQ_CLOSE_TODAY']
# Dividend yield: DTOP DPIBS
data_day_descriptor['DTOP'] = data_day_descriptor['DTOP'].fillna(0)
data_day_descriptor['DPIBS'] = data_day_descriptor['WEST_AVGDPS_FY1'] / data_day_descriptor['S_DQ_CLOSE_TODAY']
# Volatility: BETA; DASTD CMRA HSIGMA
# Growth: EGRO_ST SGRO_ST EGRO_LT SGRO_LT EGRO_MF
data_day_descriptor['EGRO_MF'] = data_day_descriptor['WEST_NETPROFIT_CAGR']
# Liquidity: STOM STOQ STOA
# Sentiment: WEST_NETPROFIT_FY1_3M WEST_SALES_FY1_3M
# Other
data_day_descriptor['WEIGHT'] = data_day_descriptor['S_DQ_MV'] * 10000

retained_columns = ['TRADE_DT', 'WIND_CODE', 'BTOP', 'ETOP', 'EPIBS', 'CETOP', 'ENMU', 'LSIZE', 'MID_CAP', 'STREV', 'RSTR', 'HALPHA', 'MLEV', 'BLEV', 'DTOA', 'VSAL', 'VERN',
                    'VFLO', 'SPIBS', 'ACBS', 'ACCF', 'CETOE', 'ATO', 'GP', 'GPM', 'ROA', 'IGRO', 'AGRO', 'CXGRO', 'DTOP', 'DPIBS', 'BETA', 'DASTD', 'CMRA', 'HSIGMA', 'EGRO_ST',
                    'SGRO_ST', 'EGRO_LT', 'SGRO_LT', 'EGRO_MF', 'STOM', 'STOQ', 'STOA', 'WEST_NETPROFIT_FY1_3M', 'WEST_SALES_FY1_3M', 'INDUSTRY_CITIC', 'WEIGHT', 'NEXT_RETURN']
loc_t = data_day_descriptor.TRADE_DT >= datetime(2009, 1, 1)
data_day_descriptor = data_day_descriptor.loc[loc_t, retained_columns]
data_day_descriptor.dropna(subset=['NEXT_RETURN'], inplace=True)
data_day_descriptor.drop_duplicates(subset=['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_descriptor.to_csv(r'D:\Risk_Model\Data\data_day_descriptor_p2.csv')


# Part IV 数据标准化处理
data_day_descriptor = pd.read_csv(r'D:\Risk_Model\Data\data_day_descriptor_p2.csv')
data_day_descriptor.drop('Unnamed: 0', axis=1, inplace=True)
data_day_descriptor.dropna(subset=['INDUSTRY_CITIC'], inplace=True)
data_day_descriptor.drop_duplicates(subset=['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_descriptor.set_index(['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_descriptor.sort_values(by=['TRADE_DT', 'WIND_CODE'], inplace=True)
# 行业_市值中性化处理；Beta_市值中性化处理；市值中性化处理
Industry_Size_NC = ['BTOP', 'ETOP', 'EPIBS', 'CETOP', 'ENMU', 'MLEV', 'BLEV', 'DTOA', 'VSAL', 'VERN', 'VFLO', 'SPIBS', 'ACBS', 'ACCF', 'CETOE',
                    'ATO', 'GP', 'GPM', 'ROA', 'IGRO', 'AGRO', 'CXGRO', 'DTOP', 'DPIBS', 'EGRO_ST', 'SGRO_ST', 'EGRO_LT', 'SGRO_LT', 'EGRO_MF']
Beta_Size_NC = ['DASTD', 'CMRA', 'HSIGMA']
Size_NC = ['STOM', 'STOQ', 'STOA']
processed_len = len(data_day_descriptor.index.get_level_values(0).unique())
for i_num, i_date in enumerate(data_day_descriptor.index.get_level_values(0).unique()):
    for j_column in Industry_Size_NC:
        data_day_descriptor_t = data_day_descriptor.loc[i_date, [j_column, 'LSIZE', 'INDUSTRY_CITIC', 'WEIGHT']].copy()
        loc_t = ~data_day_descriptor_t[j_column].isnull()
        data_day_descriptor_t2 = data_day_descriptor_t[loc_t].copy()  # t2: not null
        if data_day_descriptor_t2.shape[0] < 100:
            continue
        data_day_dummies = pd.get_dummies(data_day_descriptor_t2['INDUSTRY_CITIC'])
        data_day_descriptor_t2 = pd.merge(data_day_dummies, data_day_descriptor_t2, how='left', left_index=True, right_index=True, sort=False)
        y = data_day_descriptor_t2[j_column].values
        columns_t = list(data_day_dummies.columns) + ['LSIZE']
        x = data_day_descriptor_t2[columns_t].values
        stock_weights = data_day_descriptor_t2['WEIGHT'].values
        stock_weights = stock_weights ** 2
        wls_results = sm.WLS(y, x, weights=stock_weights).fit()  # 没有截距项
        data_day_descriptor_t.loc[loc_t, j_column] = wls_results.resid
        data_day_descriptor.loc[(i_date, slice(None)), j_column] = data_day_descriptor_t[j_column].values
    for j_column in Beta_Size_NC:
        data_day_descriptor_t = data_day_descriptor.loc[i_date, [j_column, 'BETA', 'LSIZE', 'WEIGHT']].copy()
        loc_t = (data_day_descriptor_t[j_column].isnull()) | (data_day_descriptor_t['BETA'].isnull())
        loc_t = ~loc_t
        data_day_descriptor_t2 = data_day_descriptor_t[loc_t].copy()  # t2: not null
        if data_day_descriptor_t2.shape[0] < 100:
            continue
        y = data_day_descriptor_t2[j_column].values
        columns_t = ['BETA', 'LSIZE']
        x = data_day_descriptor_t2[columns_t].values
        stock_weights = data_day_descriptor_t2['WEIGHT'].values
        stock_weights = stock_weights ** 2
        wls_results = sm.WLS(y, sm.add_constant(x), weights=stock_weights).fit()  # 加入截距项
        data_day_descriptor_t.loc[loc_t, j_column] = wls_results.resid
        data_day_descriptor.loc[(i_date, slice(None)), j_column] = data_day_descriptor_t[j_column].values
    for j_column in Size_NC:
        data_day_descriptor_t = data_day_descriptor.loc[i_date, [j_column, 'LSIZE', 'WEIGHT']].copy()
        loc_t = (data_day_descriptor_t[j_column].isnull())
        loc_t = ~loc_t
        data_day_descriptor_t2 = data_day_descriptor_t[loc_t].copy()  # t2: not null
        if data_day_descriptor_t2.shape[0] < 100:
            continue
        y = data_day_descriptor_t2[j_column].values
        x = data_day_descriptor_t2['LSIZE'].values
        stock_weights = data_day_descriptor_t2['WEIGHT'].values
        stock_weights = stock_weights ** 2
        wls_results = sm.WLS(y, sm.add_constant(x), weights=stock_weights).fit()  # 加入截距项
        data_day_descriptor_t.loc[loc_t, j_column] = wls_results.resid
        data_day_descriptor.loc[(i_date, slice(None)), j_column] = data_day_descriptor_t[j_column].values
    if i_num % 100 == 0:
        print('Part_IV_I已完成：', int(i_num*100/processed_len), '%')


def yn_std(data, weight):
    fun_data = data.copy()
    fun_weight = weight.copy()
    if fun_data.empty:  # 是否全部为空
        return fun_data
    elif fun_data.abs().sum() == 0:  # 是否全部为0
        return fun_data
    else:
        t_mean = fun_data.mean(skipna=True)
        t_std = fun_data.std(skipna=True)
        fun_data.loc[fun_data > (t_mean + (3 * t_std))] = t_mean + (3 * t_std)
        fun_data.loc[fun_data < (t_mean - (3 * t_std))] = t_mean - (3 * t_std)
        # do not fillna
        loc_t = ~fun_data.isnull()
        t_weighted_mean = np.dot(fun_data.loc[loc_t], fun_weight.loc[loc_t] / fun_weight.loc[loc_t].sum())
        fun_data = (fun_data - t_weighted_mean) / t_std
        return fun_data


def yn_nlsize(data, weight):
    fun_data = data.copy()
    loc_t = ~fun_data.isnull()
    x = data.loc[loc_t].values
    y = x ** 3
    stock_weights = weight.loc[loc_t].values
    stock_weights = stock_weights ** 2
    wls_results = sm.WLS(y, x, weights=stock_weights).fit()
    fun_data.loc[loc_t] = wls_results.resid
    return fun_data


columns_t = ['BTOP', 'ETOP', 'EPIBS', 'CETOP', 'ENMU', 'LSIZE', 'MID_CAP', 'STREV', 'RSTR', 'HALPHA', 'MLEV', 'BLEV', 'DTOA', 'VSAL', 'VERN',
             'VFLO', 'SPIBS', 'ACBS', 'ACCF', 'CETOE', 'ATO', 'GP', 'GPM', 'ROA', 'IGRO', 'AGRO', 'CXGRO', 'DTOP', 'DPIBS', 'BETA', 'DASTD', 'CMRA',
             'HSIGMA', 'EGRO_ST', 'SGRO_ST', 'EGRO_LT', 'SGRO_LT', 'EGRO_MF', 'STOM', 'STOQ', 'STOA', 'WEST_NETPROFIT_FY1_3M', 'WEST_SALES_FY1_3M']
IC_record = pd.DataFrame()

processed_len = len(data_day_descriptor.index.get_level_values(0).unique())
for i_num, i_date in enumerate(data_day_descriptor.index.get_level_values(0).unique()):
    data_day_descriptor.loc[(i_date, slice(None)), columns_t] = \
        data_day_descriptor.loc[(i_date, slice(None)), columns_t].apply(lambda x: yn_std(x, data_day_descriptor.loc[(i_date, slice(None)), 'WEIGHT']))
    # MID_CAP
    data_day_descriptor.loc[(i_date, slice(None)), ['MID_CAP']] = \
        data_day_descriptor.loc[(i_date, slice(None)), ['MID_CAP']].apply(lambda x: yn_nlsize(data_day_descriptor.loc[(i_date, slice(None)), 'LSIZE'], data_day_descriptor.loc[(i_date, slice(None)), 'WEIGHT']))
    # standardize MID_CAP
    data_day_descriptor.loc[(i_date, slice(None)), ['MID_CAP']] = \
        data_day_descriptor.loc[(i_date, slice(None)), ['MID_CAP']].apply(lambda x: yn_std(x, data_day_descriptor.loc[(i_date, slice(None)), 'WEIGHT']))
    # IC analysis
    temp_data_IC = data_day_descriptor.loc[(i_date, slice(None), slice(None))]  # a mistake but no influence
    IC_t = temp_data_IC[columns_t].corrwith(temp_data_IC.NEXT_RETURN)
    IC_t_df = pd.DataFrame(IC_t, columns=[i_date]).T
    IC_t_df['TRADE_DT'] = i_date
    IC_record = IC_record.append(IC_t_df, ignore_index=True)
    if i_num % 100 == 0:
        print('Part_IV_II已完成：', int(i_num*100/processed_len), '%')


data_day_descriptor.to_csv(r'D:\Risk_Model\Data\data_day_descriptor_p3.csv')
IC_record.to_csv(r'D:\Risk_Model\Data\IC_record.csv')

# Part V style因子的合成  data_t中有另外一个版本，该版本中 LSIZE 与 MID_CAP 分别作为style factor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import statsmodels.api as sm

# Part V
data_day_descriptor = pd.read_csv(r'D:\Risk_Model\Data\data_day_descriptor_p3.csv')

# Basic
Earnings_yield = ['ETOP', 'EPIBS', 'CETOP', 'ENMU']
Mid_term_momentum = ['RSTR', 'HALPHA']
Leverage = ['MLEV', 'BLEV', 'DTOA']  # Notice: Leverage需要取负数
Earnings_variability = ['VSAL', 'VERN', 'VFLO', 'SPIBS']  # Notice: Earnings_variability需要取负数
Earnings_quality = ['ACBS', 'ACCF', 'CETOE']  # Notice: ACBS需要取负数
Profitability = ['ATO', 'GP', 'GPM', 'ROA']
Investment_quality = ['IGRO', 'AGRO', 'CXGRO']  # Notice: Investment_quality需要取负数
Residual_volatility = ['DASTD', 'CMRA', 'HSIGMA']
# Notice: 方向变化
data_day_descriptor['ACBS'] = - data_day_descriptor['ACBS']  # Notice: ACBS已取负数

Basic_factors = [Earnings_yield, Mid_term_momentum, Leverage, Earnings_variability, Earnings_quality, Profitability, Investment_quality, Residual_volatility]
for i_basic_factor in Basic_factors:
    for j_descriptor in i_basic_factor:
        loc_t = data_day_descriptor[j_descriptor].isnull()
        data_day_descriptor.loc[loc_t, j_descriptor] = data_day_descriptor.loc[loc_t, i_basic_factor].sum(axis=1, skipna=True, min_count=1) / len(i_basic_factor)

Basic_factors_str = ['Earnings_yield', 'Mid_term_momentum', 'Leverage', 'Earnings_variability', 'Earnings_quality', 'Profitability', 'Investment_quality', 'Residual_volatility']
for i_num in range(len(Basic_factors_str)):
    data_day_descriptor[Basic_factors_str[i_num]] = data_day_descriptor[Basic_factors[i_num]].mean(axis=1, skipna=False)

# Style
Value = ['BTOP', 'Earnings_yield']
# Size = ['LSIZE', 'MID_CAP']
Momentum = ['STREV', 'Mid_term_momentum']
Quality = ['Leverage', 'Earnings_variability', 'Earnings_quality', 'Profitability', 'Investment_quality']
Dividend_yield = ['DTOP', 'DPIBS']
Volatility = ['BETA', 'Residual_volatility']
Growth = ['EGRO_ST', 'SGRO_ST', 'EGRO_LT', 'SGRO_LT', 'EGRO_MF']
Liquidity = ['STOM', 'STOQ', 'STOA']
Sentiment = ['WEST_NETPROFIT_FY1_3M', 'WEST_SALES_FY1_3M']
# Notice: 方向变化
# data_day_descriptor['MID_CAP'] = - data_day_descriptor['MID_CAP']
data_day_descriptor['Leverage'] = - data_day_descriptor['Leverage']
data_day_descriptor['Earnings_variability'] = - data_day_descriptor['Earnings_variability']
data_day_descriptor['Investment_quality'] = - data_day_descriptor['Investment_quality']
data_day_descriptor['Residual_volatility'] = - data_day_descriptor['Residual_volatility']  # Notice: 确认下Residual_volatility的正负号
#
Style_factors = [Value, Momentum, Quality, Dividend_yield, Volatility, Growth, Liquidity, Sentiment]
for i_style_factor in Style_factors:
    for j_basic in i_style_factor:
        loc_t = data_day_descriptor[j_basic].isnull()
        data_day_descriptor.loc[loc_t, j_basic] = data_day_descriptor.loc[loc_t, i_style_factor].sum(axis=1, skipna=True, min_count=1) / len(i_style_factor)

data_day_descriptor['Value'] = 0.35 * data_day_descriptor['BTOP'] + 0.65 * data_day_descriptor['Earnings_yield']
# data_day_descriptor['Size'] = 0.9 * data_day_descriptor['LSIZE'] + 0.1 * data_day_descriptor['MID_CAP']
data_day_descriptor['Momentum'] = 0.7 * data_day_descriptor['STREV'] + 0.3 * data_day_descriptor['Mid_term_momentum']  # Notice: 确认下Mid_term_momentum的正负号
data_day_descriptor['Quality'] = 0.125 * data_day_descriptor['Leverage'] + 0.125 * data_day_descriptor['Earnings_variability']\
                                 + 0.25 * data_day_descriptor['Earnings_quality'] + 0.25 * data_day_descriptor['Profitability'] + 0.25 * data_day_descriptor['Investment_quality']
data_day_descriptor['Dividend_yield'] = 0.5 * data_day_descriptor['DTOP'] + 0.5 * data_day_descriptor['DPIBS']
data_day_descriptor['Volatility'] = 0.6 * data_day_descriptor['BETA'] + 0.4 * data_day_descriptor['Residual_volatility']
data_day_descriptor['Growth'] = data_day_descriptor[Growth].mean(axis=1, skipna=False)
data_day_descriptor['Liquidity'] = 0.35 * data_day_descriptor['STOM'] + 0.35 * data_day_descriptor['STOQ'] + 0.3 * data_day_descriptor['STOA']
data_day_descriptor['Sentiment'] = data_day_descriptor[Sentiment].mean(axis=1, skipna=False)

retained_columns = ['TRADE_DT', 'WIND_CODE', 'Value', 'LSIZE', 'MID_CAP', 'Momentum', 'Quality', 'Dividend_yield', 'Volatility', 'Growth', 'Liquidity', 'Sentiment', 'INDUSTRY_CITIC', 'WEIGHT', 'NEXT_RETURN']
data_day_style = data_day_descriptor.loc[:, retained_columns].copy()
data_day_style.TRADE_DT = pd.to_datetime(data_day_style.TRADE_DT)
data_day_style.set_index(['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.sort_values(by=['TRADE_DT', 'WIND_CODE'], inplace=True)


def yn_std(data, weight):
    fun_data = data.copy()
    fun_weight = weight.copy()
    if fun_data.empty:  # 是否全部为空
        return fun_data
    elif fun_data.abs().sum() == 0:  # 是否全部为0
        return fun_data
    else:
        t_mean = fun_data.mean(skipna=True)
        t_std = fun_data.std(skipna=True)
        fun_data.loc[fun_data > (t_mean + (3 * t_std))] = t_mean + (3 * t_std)
        fun_data.loc[fun_data < (t_mean - (3 * t_std))] = t_mean - (3 * t_std)
        # do not fillna
        loc_t = ~fun_data.isnull()
        t_weighted_mean = np.dot(fun_data.loc[loc_t], fun_weight.loc[loc_t] / fun_weight.loc[loc_t].sum())
        fun_data = (fun_data - t_weighted_mean) / t_std
        return fun_data


columns_t = ['Value', 'LSIZE', 'MID_CAP', 'Momentum', 'Quality', 'Dividend_yield', 'Volatility', 'Growth', 'Liquidity', 'Sentiment']
IC_record_style = pd.DataFrame()
processed_len = len(data_day_style.index.get_level_values(0).unique())
for i_num, i_date in enumerate(data_day_style.index.get_level_values(0).unique()):
    data_day_style.loc[(i_date, slice(None)), columns_t] = \
        data_day_style.loc[(i_date, slice(None)), columns_t].apply(lambda x: yn_std(x, data_day_style.loc[(i_date, slice(None)), 'WEIGHT']))
    # IC
    temp_data_IC = data_day_style.loc[(i_date, slice(None))]
    IC_t = temp_data_IC[columns_t].corrwith(temp_data_IC.NEXT_RETURN)
    IC_t_df = pd.DataFrame(IC_t, columns=[i_date]).T
    IC_t_df['TRADE_DT'] = i_date
    IC_record_style = IC_record_style.append(IC_t_df, ignore_index=True)
    if i_num % 100 == 0:
        print('Part_V已完成：', int(i_num*100/processed_len), '%')

data_day_style.dropna(inplace=True)
data_day_style.to_csv(r'D:\Risk_Model\Data\data_day_style_V2.csv')

