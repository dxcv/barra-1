#coding=utf8


db_multi_factor_uri = 'mysql://zhaoliyuan:zhaoliyuan20d@1101075@192.168.88.254/multi_factor?charset=utf8&use_unicode=1'
db_base_uri =  'mysql://zhaoliyuan:zhaoliyuan20d@1101075@192.168.88.254/mofang?charset=utf8&use_unicode=1'
db_asset_uri =  'mysql://zhaoliyuan:zhaoliyuan20d@1101075@192.168.88.254/asset_allocation?charset=utf8&use_unicode=1'
db_wind_uri = 'mysql://public:h76zyeTfVqAehr5J@192.168.88.11/wind?charset=utf8&use_unicode=1'

uris = {
    'asset': db_asset_uri,
    'base': db_base_uri,
    'wind': db_wind_uri,
    'multi_factor': db_multi_factor_uri,
}

