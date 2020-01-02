import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime

# Part I regression
data_day_style = pd.read_csv(r'D:\Risk_Model\Data\data_day_style_V3.csv')  # need to change
data_day_style['TRADE_DT'] = pd.to_datetime(data_day_style['TRADE_DT'])
data_day_style.drop_duplicates(subset=['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.set_index(['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.sort_values(by=['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.dropna(inplace=True)
data_day_style['resid'] = np.nan  # record specific risk
data_day_style.rename(columns={'INDUSTRY_CITIC': 'INDUSTRY'}, inplace=True)

style = ['BTOP', 'Earnings_yield', 'LSIZE', 'MID_CAP', 'STREV', 'Quality', 'Dividend_yield', 'BETA', 'Residual_volatility', 'Growth_ST', 'Growth_LT', 'Liquidity', 'Sentiment']  # raw326
industry = ['银行', '房地产', '医药', '餐饮旅游', '商贸零售', '机械', '建材', '家电', '纺织服装', '食品饮料',
            '电子元器件', '汽车', '轻工制造', '电力及公用事业', '综合', '通信', '石油石化', '有色金属', '农林牧渔',
            '建筑', '计算机', '交通运输', '基础化工', '煤炭', '电力设备', '钢铁', '国防军工', '非银行金融', '传媒']
columns_t = ['Country'] + industry + style
params_record = pd.DataFrame(columns=columns_t)
tvalues_record = pd.DataFrame(columns=columns_t)
vif_record = pd.DataFrame()
index_t = list(data_day_style.index.get_level_values(0).unique())
R_record = pd.DataFrame(columns=['R_square'], index=index_t)
processed_len = len(data_day_style.index.get_level_values(0).unique())
for i_num, i_date in enumerate(data_day_style.index.get_level_values(0).unique()):
    data_day_style_t = data_day_style.loc[i_date].copy()
    data_day_dummies = pd.get_dummies(data_day_style_t.INDUSTRY)
    industry_t = list(data_day_dummies.columns)
    data_day_style_t = pd.merge(data_day_dummies, data_day_style_t, left_index=True, right_index=True, sort=False)
    fun_columns_t = industry_t + style
    X = sm.add_constant(data_day_style_t[fun_columns_t].values)
    Y = data_day_style_t.NEXT_RETURN.values
    matrix_A = np.mat(data_day_dummies[industry_t].T)  # 已转置处理
    matrix_B = np.mat(data_day_style_t['WEIGHT'].values.reshape(-1, 1))
    matrix_C = matrix_A * matrix_B
    industry_weights = np.array(matrix_C.T)[0] / np.array(matrix_C.T)[0].sum()
    C_R = np.hstack([[0], industry_weights, np.zeros(len(style))])
    C_Q = 0
    stock_weights = np.sqrt(data_day_style_t['WEIGHT'].values)  # np.sqrt()
    glm_model = sm.GLM(Y, X, var_weights=stock_weights)
    glm_results = glm_model.fit_constrained((C_R, C_Q))
    #
    fun_columns_t2 = ['Country'] + fun_columns_t
    params_dict = dict(zip(fun_columns_t2, glm_results.params))
    params_record = params_record.append(pd.DataFrame(params_dict, index=[i_date]), ignore_index=False, sort=False)
    #
    tvalues_dict = dict(zip(fun_columns_t2, glm_results.tvalues))
    tvalues_record = tvalues_record.append(pd.DataFrame(tvalues_dict, index=[i_date]), ignore_index=False, sort=False)
    ssr = np.dot(stock_weights, (glm_results.resid_response ** 2))
    uncentered_tss = np.dot(stock_weights, Y ** 2)
    R_record.at[i_date, 'R_square'] = 1 - ssr/uncentered_tss
    # record special risk
    data_day_style.loc[(i_date, slice(None)), 'resid'] = glm_results.resid_response
    if i_num % 100 == 0:
        print('regression已完成：', int(i_num * 100 / processed_len), '%')


# Part II
params_record.sort_index(inplace=True)
H_L_vol = 84  # factor volatility half life
H_L_corr = 252  # factor volatility half life
H_L_VRA = 42  # factor
H_L_special = 84
H_L_special_NW = 252
H_L_special_VRA = 42
Lags_vol = 5  # Newey-West volatility lags
Lags_corr = 2  # Newey-West volatility correlation
Lags_special = 5
H_window = 252  # 过去252个交易日的数据，最好使用过去两年的交易数据，因为H_L_corr=252
F_vol_raw = pd.DataFrame()
F_corr_raw = pd.DataFrame()
Predict_period = 21  # 对未来21天的风险进行预测，若低于21可能会出现问题，Newey_West处理所导致
adj_coef = 1.2  # yn_eig_risk_adj函数 方正证券取值1.2 Barra取值1.4
M = 1000  # 模拟次数
E_0 = 1.05  # E_0是一个略大于1的常数，用于去除残差项的指数次幂带来的偏误


def yn_f_raw(data, half_life):  # type of data is np.array
    fun_data = data.copy()
    lambda_t = 0.5 ** (1 / half_life)
    weights_t = lambda_t ** (np.arange(fun_data.shape[0] - 1, -1, -1))
    weights_t = weights_t / weights_t.sum()
    f_raw = np.cov(fun_data, rowvar=False, aweights=weights_t, ddof=0)  # ddof=0: 样本方差 /n 默认为: ddof=1
    return f_raw


def yn_newey_west(data, half_life, lags):
    fun_data = data.copy()
    lambda_t = 0.5 ** (1 / half_life)
    c_newey_west = np.zeros((fun_data.shape[1], fun_data.shape[1]))
    for i_lag in range(1, lags+1):
        c_newey_west_t = np.zeros((fun_data.shape[1], fun_data.shape[1]))
        for j_factor in range(fun_data.shape[1]):
            fun_data_t = fun_data.copy()  # Notice: use copy
            fun_data_j = fun_data_t[:-i_lag, j_factor]
            fun_data_t = fun_data_t[i_lag:, :]
            weights_t = lambda_t ** (np.arange(fun_data_t.shape[0] - 1, -1, -1))
            weights_t = weights_t / weights_t.sum()
            volatility_t = np.cov(fun_data_t, fun_data_j, rowvar=False, aweights=weights_t, ddof=0)
            c_newey_west_t[:, j_factor] = volatility_t[:-1, -1]
        coef_t = 1 - (i_lag / (lags + 1))
        c_newey_west = c_newey_west + coef_t * (c_newey_west_t + c_newey_west_t.T)
    return c_newey_west


def yn_eig_risk_adj(data):  # 会使用全局变量: M H_window adj_coef
    f_nw = data.copy()
    w_0, u_0 = np.linalg.eig(f_nw)
    d_0 = np.mat(u_0.T) * np.mat(f_nw) * np.mat(u_0)
    m_volatility = np.zeros((M, f_nw.shape[0]))
    # 模拟M次
    for m in range(M):
        b_m = np.zeros((f_nw.shape[0], H_window))  # N*T
        for i_row in range(b_m.shape[0]):
            b_m[i_row, :] = np.random.normal(loc=0, scale=np.sqrt(d_0[i_row, i_row]), size=H_window)
        r_m = np.mat(u_0) * np.mat(b_m)
        r_m = np.array(r_m.T)  # notice: 转置处理
        f_nw_m = np.cov(r_m, rowvar=False, ddof=0)  # 不需要Weights
        w_m, u_m = np.linalg.eig(f_nw_m)
        d_m = np.mat(u_m.T) * np.mat(f_nw_m) * np.mat(u_m)
        d_m_real = np.mat(u_m.T) * np.mat(f_nw) * np.mat(u_m)
        m_volatility[m, :] = np.diag(d_m_real) / np.diag(d_m)
    gamma_t = np.sqrt(m_volatility.mean(axis=0))
    gamma_t = adj_coef * (gamma_t - 1) + 1
    return gamma_t


def yn_vol_regime_adj(data):  # 会使用全局变量: H_L_VRA
    fun_data = data[:-1]
    fun_data = fun_data[~np.isnan(fun_data)]
    if len(fun_data) < H_L_VRA:
        return np.nan
    else:
        lambda_t = 0.5 ** (1 / H_L_VRA)
        weights_t = lambda_t ** (np.arange(fun_data.shape[0] - 1, -1, -1))
        weights_t = weights_t / weights_t.sum()
        lambda_f_var = np.dot(fun_data ** 2, weights_t)
        lambda_f = np.sqrt(lambda_f_var)
        return lambda_f


# 先计算波动率偏误，因为Predict_period_VRA = 1，所以无需进行Newey_West调整
Predict_period_VRA = 1
params_record['instantaneous_bias'] = np.nan
columns_t = list(params_record.columns)
columns_t.remove('Country')
columns_t.remove('instantaneous_bias')
processed_len = len(params_record.index)
for i_num, i_date in enumerate(params_record.index):
    if i_num < H_window:
        continue
    data_t = params_record.iloc[(i_num - H_window):i_num][columns_t].copy()  # exclude the data of i_date
    data_t.dropna(axis=1, thresh=0.8 * data_t.shape[0], inplace=True)  # 数据完整度80%以上
    data_t.fillna(0, inplace=True)
    data_t_array = data_t.values
    # volatility
    F_raw_vol = yn_f_raw(data=data_t_array, half_life=H_L_vol)  # volatility_raw
    # correlation
    F_raw_corr = yn_f_raw(data=data_t_array, half_life=H_L_corr)  # volatility_raw
    # combine the volatility and correlation
    F_raw = F_raw_corr.copy()
    for i in range(F_raw.shape[0]):
        for j in range(F_raw.shape[1]):
            F_raw[i, j] = F_raw[i, j] / np.sqrt(F_raw_corr[i, i] * F_raw_corr[j, j]) * np.sqrt(F_raw_vol[i, i] * F_raw_vol[j, j])
    F_raw = Predict_period_VRA * F_raw  # One day
    # Eigenfactor Risk Adjustment
    gamma_array = yn_eig_risk_adj(data=F_raw)
    gamma_matrix = np.mat(np.diag(gamma_array**2))
    W_0, U_0 = np.linalg.eig(F_raw)
    D_0 = np.mat(U_0.T) * np.mat(F_raw) * np.mat(U_0)
    D_real = gamma_matrix * D_0
    F_eigen = np.mat(U_0) * D_real * np.mat(U_0.T)
    # record the instantaneous_bias
    t_risk = np.sqrt(np.diag(F_eigen))  # notice
    t_return = params_record.iloc[i_num][data_t.columns].fillna(0).values  # 未来收益与预测风险
    t_std_return = t_return / t_risk
    B_F = np.sqrt(np.mean(t_std_return ** 2))
    params_record.at[i_date, 'instantaneous_bias'] = B_F
    if i_num % 100 == 0:
        print('Part_II_I已完成：', int(i_num*100/processed_len), '%')

params_record['VRA_adj'] = params_record['instantaneous_bias'].rolling(H_window+1, min_periods=H_L_VRA+1).apply(lambda x: yn_vol_regime_adj(x), 'raw=True')

factor_risk_record = pd.DataFrame()
processed_len = len(params_record.index)
for i_num, i_date in enumerate(params_record.index):
    if i_num < H_window:
        continue
    data_t = params_record.iloc[(i_num - H_window):i_num][columns_t].copy()  # exclude i_date
    data_t.dropna(axis=1, thresh=0.8 * data_t.shape[0], inplace=True)  # 数据完整度80%以上
    data_t.fillna(0, inplace=True)
    data_t_array = data_t.values
    # volatility
    F_raw_vol = yn_f_raw(data=data_t_array, half_life=H_L_vol)  # volatility_raw
    C_Newey_West_vol = yn_newey_west(data=data_t_array, half_life=H_L_vol, lags=Lags_vol)  # Newey_West
    F_NW_vol = F_raw_vol + C_Newey_West_vol  # volatility_Newey_West
    # correlation
    F_raw_corr = yn_f_raw(data=data_t_array, half_life=H_L_corr)  # volatility_raw
    C_Newey_West_corr = yn_newey_west(data=data_t_array, half_life=H_L_corr, lags=Lags_corr)  # Newey_West
    F_NW_corr = F_raw_corr + C_Newey_West_corr  # volatility_Newey_West
    # combine
    F_NW = F_NW_corr.copy()
    for i in range(F_NW.shape[0]):
        for j in range(F_NW.shape[1]):
            F_NW[i, j] = F_NW[i, j] / np.sqrt(F_NW_corr[i, i] * F_NW_corr[j, j]) * np.sqrt(F_NW_vol[i, i] * F_NW_vol[j, j])
    F_NW = Predict_period * F_NW  # 日频方差--->月频方差
    # Eigenfactor Risk Adjustment
    gamma_array = yn_eig_risk_adj(data=F_NW)
    gamma_matrix = np.mat(np.diag(gamma_array**2))
    W_0, U_0 = np.linalg.eig(F_NW)
    D_0 = np.mat(U_0.T) * np.mat(F_NW) * np.mat(U_0)
    D_real = gamma_matrix * D_0
    F_eigen = np.mat(U_0) * D_real * np.mat(U_0.T)
    # Volatility regime adjustment
    lambda_VRA = params_record.at[i_date, 'VRA_adj']
    F_VRA = (lambda_VRA ** 2) * F_eigen
    df_t = pd.DataFrame(F_VRA, columns=data_t.columns, index=range(F_VRA.shape[0]))
    df_t['FACTOR'] = list(data_t.columns)
    df_t['TRADE_DT'] = i_date
    factor_risk_record = factor_risk_record.append(df_t, ignore_index=True, sort=False)
    if i_num % 100 == 0:
        print('Part_II_II已完成：', int(i_num*100/processed_len), '%')


factor_risk_record.to_csv(r'D:\Risk_Model\factor_risk_record.csv')

# Part III

data_day_style.reset_index(inplace=True)
data_day_style.set_index(['WIND_CODE'], inplace=True)  # 单索引
data_day_style.sort_values(by=['WIND_CODE', 'TRADE_DT'], inplace=True)
data_day_style['specific_risk_raw'] = np.nan
data_day_style['specific_risk_NW'] = np.nan
data_day_style['coordination_coef'] = np.nan
data_day_style['specific_risk_SM_1'] = np.nan
data_day_style['specific_risk_shrink_1'] = np.nan
data_day_style['specific_risk_SM_Predict'] = np.nan
data_day_style['specific_risk_shrink_Predict'] = np.nan
data_day_style['specific_risk_VRA_Predict'] = np.nan


def yn_specific_raw(data):  # 返回std; 会使用全局变量: H_L_special
    fun_data = data[:-1]
    fun_data = fun_data[~np.isnan(fun_data)]
    if len(fun_data) < H_L_special:
        return np.nan
    else:
        lambda_t = 0.5 ** (1 / H_L_special)
        weights_t = lambda_t ** (np.arange(len(fun_data) - 1, -1, -1))
        weights_t = weights_t / weights_t.sum()
        var_raw = np.cov(fun_data, aweights=weights_t)  # type(avr_raw): array
        return np.sqrt(float(var_raw))


def yn_specific_nw(data):  # 会使用全局变量: H_L_special Lags_special
    fun_data = data[:-1]
    fun_data = fun_data[~np.isnan(fun_data)]
    if len(fun_data) < H_L_special:
        return np.nan
    else:
        lambda_t = 0.5 ** (1 / H_L_special)
        weights_t = lambda_t ** (np.arange(len(fun_data) - 1, -1, -1))
        weights_t = weights_t / weights_t.sum()
        var_raw = np.cov(fun_data, aweights=weights_t)  # type(avr_raw): array
        c_nw = 0
        for k_lag in range(1, Lags_special + 1):
            data_c1 = fun_data[k_lag:]
            data_c2 = fun_data[:-k_lag]
            lambda_tc = 0.5 ** (1 / H_L_special_NW)
            weights_tc = lambda_tc ** (np.arange(len(data_c1) - 1, -1, -1))
            c_t = np.cov(data_c1, data_c2, aweights=weights_tc)
            coef_t = 1 - (k_lag / (Lags_special + 1))
            c_nw = c_nw + coef_t * 2 * c_t[0, 1]
        var_nw = float(var_raw) + c_nw
        return np.sqrt(Predict_period * var_nw)


def yn_robust_std(data):  # 会使用全局变量: H_window
    fun_data = data[:-1]
    fun_data = fun_data[~np.isnan(fun_data)]
    if len(fun_data) < H_L_special:
        return np.nan
    else:
        q_1, q_3 = np.percentile(fun_data, [25, 75])
        sigma_u = (1/1.35) * (q_3 - q_1)
        z_u = (np.std(fun_data) - sigma_u) / sigma_u
        z_u = np.abs(z_u)
        gamma_t = min(1, max(0, (H_window-60)/120)) * min(1, max(0, np.exp(1-z_u)))
        return gamma_t


def yn_vol_regime_adj2(data):
    fun_data = data[:-1]
    fun_data = fun_data[~np.isnan(fun_data)]
    if len(fun_data) < H_L_special_VRA:
        return np.nan
    else:
        lambda_t = 0.5 ** (1 / H_L_special_VRA)
        weights_t = lambda_t ** (np.arange(fun_data.shape[0] - 1, -1, -1))
        weights_t = weights_t / weights_t.sum()
        lambda_f_var = np.dot(fun_data ** 2, weights_t)
        lambda_f = np.sqrt(lambda_f_var)
        return lambda_f


# 先进行波动率偏误的计算
processed_len = len(data_day_style.index.get_level_values(0).unique())
for i_num, i_code in enumerate(data_day_style.index.get_level_values(0).unique()):
    data_day_style.loc[i_code, 'specific_risk_raw'] = data_day_style.loc[i_code, 'resid'].rolling(H_window+1).apply(lambda x: yn_specific_raw(x), 'raw=True')
    data_day_style.loc[i_code, 'coordination_coef'] = data_day_style.loc[i_code, 'resid'].rolling(H_window+1).apply(lambda x: yn_robust_std(x), 'raw=True')
    data_day_style.loc[i_code, 'specific_risk_NW'] = data_day_style.loc[i_code, 'resid'].rolling(H_window+1).apply(lambda x: yn_specific_nw(x), 'raw=True')
    if i_num % 100 == 0:
        print('Part_III_I已完成：', int(i_num*100/processed_len), '%')

# Notice: specific_risk_raw 为未来1天的风险，specific_risk_NW 为未来predict_period天的风险
data_day_style['coordination_coef'] = data_day_style['coordination_coef'].fillna(0)  # notice
data_day_style.loc[data_day_style.coordination_coef == 0, 'specific_risk_raw'] = 0

data_day_style.reset_index(inplace=True)
data_day_style.set_index(['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.sort_values(by=['TRADE_DT', 'WIND_CODE'], inplace=True)
VRA_bias = pd.DataFrame()
# style = ['BTOP', 'Earnings_yield', 'LSIZE', 'MID_CAP', 'STREV', 'Quality', 'Dividend_yield', 'Volatility', 'Growth', 'Liquidity', 'Sentiment']
# Assumption: 股票的本期因子暴露 ≈ 上期因子暴露
processed_len = len(data_day_style.index.get_level_values(0).unique())
SM_percent_record = list()
for i_num, i_date in enumerate(data_day_style.index.get_level_values(0).unique()):
    data_day_style_t = data_day_style.loc[i_date].copy()
    loc_t = (data_day_style_t['coordination_coef'] == 1) & (data_day_style_t['specific_risk_raw'] != 0)
    SM_percent_record.append(data_day_style_t.loc[loc_t].shape[0] / data_day_style_t.shape[0])
    if data_day_style_t.loc[loc_t].shape[0] < 0.5*data_day_style_t.shape[0]:
        continue
    else:
        data_day_dummies = pd.get_dummies(data_day_style_t.INDUSTRY)
        data_day_style_t = pd.merge(data_day_style_t, data_day_dummies, how='left', left_index=True, right_index=True, sort=False)
        industry_t = list(data_day_style_t.loc[loc_t, 'INDUSTRY'].unique())
        columns_t = industry_t + style
        x = data_day_style_t.loc[loc_t, columns_t].values
        X = sm.add_constant(x)
        y = data_day_style_t.loc[loc_t, 'specific_risk_raw'].values   # notice: 写下一个相同的循环时，需修改此处 specific_risk_NW
        Y = np.log(y)
        stock_weights = data_day_style_t.loc[loc_t, 'WEIGHT'].values
        stock_weights = np.sqrt(stock_weights)
        wls_model = sm.WLS(Y, X, weights=stock_weights)  # Notice: stock_weights
        wls_results = wls_model.fit()
        params_t = wls_results.params
        #
        x_predict = data_day_style_t.loc[~loc_t, columns_t].values
        X_predict = sm.add_constant(x_predict)
        Y_predict = np.mat(X_predict) * np.mat(params_t.reshape(-1, 1))
        y_predict = np.array(Y_predict.T)[0]
        y_predict = E_0 * np.exp(y_predict)
        #
        data_day_style_t.loc[loc_t, 'specific_risk_SM_1'] = data_day_style_t.loc[loc_t, 'specific_risk_raw']
        specific_risk_t = data_day_style_t.loc[~loc_t, 'coordination_coef'].values * data_day_style_t.loc[~loc_t, 'specific_risk_raw'].values
        structure_risk_t = (1 - data_day_style_t.loc[~loc_t, 'coordination_coef'].values) * y_predict
        data_day_style_t.loc[~loc_t, 'specific_risk_SM_1'] = specific_risk_t + structure_risk_t
        data_day_style.loc[(i_date, slice(None)), 'specific_risk_SM_1'] = data_day_style_t['specific_risk_SM_1'].values
        # Bayesian Shrinkage
        data_day_style_t = data_day_style.loc[i_date].copy()
        data_day_style_t['group'] = pd.qcut(data_day_style_t.WEIGHT, 10, labels=range(10))
        for i_group in range(10):
            loc_t = data_day_style_t['group'] == i_group
            sigma_wm = np.dot(data_day_style_t.loc[loc_t, 'specific_risk_SM_1'], data_day_style_t.loc[loc_t, 'WEIGHT'] / data_day_style_t.loc[loc_t, 'WEIGHT'].sum())
            q_shrink = 1
            t_shrink = (data_day_style_t.loc[loc_t, 'specific_risk_SM_1'].values - sigma_wm)
            delta_shrink = np.sqrt(np.mean(t_shrink ** 2))
            v_shrink = q_shrink * np.abs(t_shrink) / (delta_shrink + q_shrink * np.abs(t_shrink))
            sigma_shrink = (v_shrink * sigma_wm) + ((1 - v_shrink) * data_day_style_t.loc[loc_t, 'specific_risk_SM_1'].values)
            data_day_style_t.loc[loc_t, 'specific_risk_shrink_1'] = sigma_shrink
        data_day_style.loc[(i_date, slice(None)), 'specific_risk_shrink_1'] = data_day_style_t['specific_risk_shrink_1'].values
        # volatility regime adjustment: step 1
        data_day_style_t = data_day_style.loc[i_date].copy()
        std_specific_return = (data_day_style_t['resid'] / data_day_style_t['specific_risk_shrink_1']).values
        weights_t = data_day_style_t['WEIGHT'].values / data_day_style_t['WEIGHT'].sum()
        VRA_bias_t = np.sqrt(np.dot(weights_t, std_specific_return ** 2))
        dict_t = {'instantaneous_bias': VRA_bias_t, 'TRADE_DT': i_date}
        VRA_bias = VRA_bias.append(pd.DataFrame(dict_t, index=[0]), ignore_index=True, sort=False)
        if i_num % 100 == 0:
            print('Part_III_II已完成：', int(i_num * 100 / processed_len), '%')


VRA_bias.sort_values(by='TRADE_DT', inplace=True)
VRA_bias['VRA_adj'] = VRA_bias['instantaneous_bias'].rolling(H_window+1, min_periods=H_L_special_VRA+1).apply(lambda x: yn_vol_regime_adj2(x), 'raw=True')

###############################################
data_day_style.reset_index(inplace=True)
data_day_style = pd.merge(data_day_style, VRA_bias, how='left', on='TRADE_DT', sort=False)
data_day_style.set_index(['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.sort_values(by=['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.loc[data_day_style.coordination_coef == 0, 'specific_risk_NW'] = 0

processed_len = len(data_day_style.index.get_level_values(0).unique())
for i_num, i_date in enumerate(data_day_style.index.get_level_values(0).unique()):
    data_day_style_t = data_day_style.loc[i_date].copy()

    loc_t = (data_day_style_t['coordination_coef'] == 1) & (data_day_style_t['specific_risk_NW'] != 0)
    if data_day_style_t.loc[loc_t].shape[0] < 0.5*data_day_style_t.shape[0]:
        continue
    else:
        data_day_dummies = pd.get_dummies(data_day_style_t.INDUSTRY)
        data_day_style_t = pd.merge(data_day_style_t, data_day_dummies, how='left', left_index=True, right_index=True, sort=False)
        industry_t = list(data_day_style_t.loc[loc_t, 'INDUSTRY'].unique())
        columns_t = industry_t + style
        x = data_day_style_t.loc[loc_t, columns_t].values
        X = sm.add_constant(x)
        y = data_day_style_t.loc[loc_t, 'specific_risk_NW'].values  # Notice: specific_risk_NW
        Y = np.log(y)
        stock_weights = data_day_style_t.loc[loc_t, 'WEIGHT'].values
        stock_weights = np.sqrt(stock_weights)
        wls_model = sm.WLS(Y, X, weights=stock_weights)
        wls_results = wls_model.fit()
        params_t = wls_results.params
        #
        x_predict = data_day_style_t.loc[~loc_t, columns_t].values
        X_predict = sm.add_constant(x_predict)
        Y_predict = np.mat(X_predict) * np.mat(params_t.reshape(-1, 1))
        y_predict = np.array(Y_predict.T)[0]
        y_predict = E_0 * np.exp(y_predict)
        #
        data_day_style_t.loc[loc_t, 'specific_risk_SM_Predict'] = data_day_style_t.loc[loc_t, 'specific_risk_NW']
        specific_risk_t = data_day_style_t.loc[~loc_t, 'coordination_coef'].values * data_day_style_t.loc[~loc_t, 'specific_risk_NW'].values
        structure_risk_t = (1 - data_day_style_t.loc[~loc_t, 'coordination_coef'].values) * y_predict
        data_day_style_t.loc[~loc_t, 'specific_risk_SM_Predict'] = specific_risk_t + structure_risk_t
        data_day_style.loc[(i_date, slice(None)), 'specific_risk_SM_Predict'] = data_day_style_t['specific_risk_SM_Predict'].values
        # Bayesian Shrinkage
        data_day_style_t = data_day_style.loc[i_date].copy()
        data_day_style_t['group'] = pd.qcut(data_day_style_t.WEIGHT, 10, labels=range(10))
        for i_group in range(10):
            loc_t = data_day_style_t['group'] == i_group
            sigma_wm = np.dot(data_day_style_t.loc[loc_t, 'specific_risk_SM_Predict'], data_day_style_t.loc[loc_t, 'WEIGHT'] / data_day_style_t.loc[loc_t, 'WEIGHT'].sum())
            q_shrink = 1
            t_shrink = (data_day_style_t.loc[loc_t, 'specific_risk_SM_Predict'].values - sigma_wm)
            delta_shrink = np.sqrt(np.mean(t_shrink ** 2))
            v_shrink = q_shrink * np.abs(t_shrink) / (delta_shrink + q_shrink * np.abs(t_shrink))
            sigma_shrink = (v_shrink * sigma_wm) + ((1 - v_shrink) * data_day_style_t.loc[loc_t, 'specific_risk_SM_Predict'].values)
            data_day_style_t.loc[loc_t, 'specific_risk_shrink_Predict'] = sigma_shrink
        data_day_style.loc[(i_date, slice(None)), 'specific_risk_shrink_Predict'] = data_day_style_t['specific_risk_shrink_Predict'].values
        data_day_style.loc[(i_date, slice(None)), 'specific_risk_VRA_Predict'] = data_day_style.loc[(i_date, slice(None)), 'specific_risk_shrink_Predict'] * data_day_style.loc[(i_date, slice(None)), 'VRA_adj']
        if i_num % 100 == 0:
            print('Part_III_III已完成：', int(i_num * 100 / processed_len), '%')

# data_day_style.reset_index(inplace=True)
# columns_t = ['TRADE_DT', 'WIND_CODE', 'specific_risk_VRA_Predict']
# specific_risk_record = data_day_style[columns_t]
# specific_risk_record.to_csv(r'D:\Risk_Model\specific_risk_record.csv')
data_day_style.to_csv(r'D:\Risk_Model\specific_risk_record.csv')
