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
# Value = ['BTOP', 'Earnings_yield']
# Size = ['LSIZE', 'MID_CAP']
# Momentum = ['STREV', 'Mid_term_momentum']
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
Style_factors = [Quality, Dividend_yield, Volatility, Growth, Liquidity, Sentiment]
for i_style_factor in Style_factors:
    for j_basic in i_style_factor:
        loc_t = data_day_descriptor[j_basic].isnull()
        data_day_descriptor.loc[loc_t, j_basic] = data_day_descriptor.loc[loc_t, i_style_factor].sum(axis=1, skipna=True, min_count=1) / len(i_style_factor)

# data_day_descriptor['Value'] = 0.35 * data_day_descriptor['BTOP'] + 0.65 * data_day_descriptor['Earnings_yield']
# data_day_descriptor['Size'] = 0.9 * data_day_descriptor['LSIZE'] + 0.1 * data_day_descriptor['MID_CAP']
# data_day_descriptor['Momentum'] = 0.7 * data_day_descriptor['STREV'] + 0.3 * data_day_descriptor['Mid_term_momentum']  # Notice: 确认下Mid_term_momentum的正负号
data_day_descriptor['Quality'] = 0.125 * data_day_descriptor['Leverage'] + 0.125 * data_day_descriptor['Earnings_variability']\
                                 + 0.25 * data_day_descriptor['Earnings_quality'] + 0.25 * data_day_descriptor['Profitability'] + 0.25 * data_day_descriptor['Investment_quality']
data_day_descriptor['Dividend_yield'] = 0.5 * data_day_descriptor['DTOP'] + 0.5 * data_day_descriptor['DPIBS']
data_day_descriptor['Volatility'] = 0.6 * data_day_descriptor['BETA'] + 0.4 * data_day_descriptor['Residual_volatility']
data_day_descriptor['Growth'] = data_day_descriptor[Growth].mean(axis=1, skipna=False)
data_day_descriptor['Liquidity'] = 0.35 * data_day_descriptor['STOM'] + 0.35 * data_day_descriptor['STOQ'] + 0.3 * data_day_descriptor['STOA']
data_day_descriptor['Sentiment'] = data_day_descriptor[Sentiment].mean(axis=1, skipna=False)

retained_columns = ['TRADE_DT', 'WIND_CODE', 'BTOP', 'Earnings_yield', 'LSIZE', 'MID_CAP', 'STREV', 'Quality', 'Dividend_yield', 'Volatility', 'Growth', 'Liquidity', 'Sentiment', 'INDUSTRY_CITIC', 'WEIGHT', 'NEXT_RETURN']
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


columns_t = ['Quality', 'Dividend_yield', 'Volatility', 'Growth', 'Liquidity', 'Sentiment']  # 'LSIZE', 'MID_CAP', 'STREV'
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
data_day_style.to_csv(r'D:\Risk_Model\Data\data_day_style_V3.csv')
# IC_record_style.to_csv(r'D:\Risk_Model\Data\IC_record.csv')







































