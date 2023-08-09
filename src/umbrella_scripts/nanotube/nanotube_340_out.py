from umbrella_sampling import CustomObservableUmbrellaSampling
from oxdna_simulation import SimulationManager
import os
import matplotlib.pyplot as plt
import timeit


def main(): 
    path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/nanotube/in_out')
    conditions = ['340']
    system_name = ['out']
    file_dirs = []
    for cond in conditions:
        for sys in system_name:
            file_dirs.append(f'{path}/{cond}/{sys}/equlibration_2')
    # file_dirs = [f'{path}/{sys}' for sys in system_name]
    systems = [f'umbrella_1' for _ in range(len(file_dirs))]
    
    com_list = '3566,3774,3567,3568,3569,3570,3571,3572,3573,3574,3575,3576,3577,3578,3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3590,3591,3592,3593,3594,3595,3596,3597,3598,3599,3600,3601,3602,3603,3604,3605,3606,3607,3608,3609,3610,3611,3612,3613,3614,3615,3616,3617,3618,3619,3620,3621,3622,3623,3624,3625,3626,3627,3628,3629,3630,3631,3632,3633,3634,3635,3636,3637,3638,3639,3640,3641,3642,3643,3644,3645,3646,3647,3648,3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,3667,3668,3669,3670,3671,3672,3673,3674,3675,3676,3677,3678,3679,3680,3681,3682,3683,3684,3685,3686,3687,3688,3689,3690,3691,3692,3693,3694,3695,3696,3697,3698,3699,3700,3701,3702,3703,3704,3705,3706,3707,3708,3709,3710,3711,3712,3713,3714,3715,3716,3717,3718,3719,3720,3721,3722,3723,3724,3725,3726,3727,3728,3729,3730,3731,3732,3733,3734,3735,3736,3737,3738,3739,3740,3741,3742,3743,3744,3745,3746,3747,3748,3749,3750,3751,3752,3753,3754,3755,3756,3757,3758,3759,3760,3761,3762,3763,3764,3765,3766,3767,3768,3769,3770,3771,3772,3773'
    ref_list = '7163,6966,6967,6968,6969,6970,6971,6972,6973,6974,6975,6976,6977,6978,6979,6980,6981,6982,6983,6984,6985,6986,6987,6988,6989,6990,6991,6992,6993,6994,6995,6996,6997,6998,6999,7000,7001,7002,7003,7004,7005,7006,7007,7008,7009,7010,7011,7012,7013,7014,7015,7016,7017,7018,7019,7020,7021,7022,7023,7024,7025,7026,7027,7028,7029,7030,7031,7032,7033,7034,7035,7036,7037,7038,7039,7040,7041,7042,7043,7044,7045,7046,7047,7048,7049,7050,7051,7052,7053,7054,7055,7056,7057,7058,7059,7060,7061,7062,7063,7064,7065,7066,7067,7068,7069,7070,7071,7072,7073,7074,7075,7076,7077,7078,7079,7080,7081,7082,7083,7084,7085,7086,7087,7088,7089,7090,7091,7092,7093,7094,7095,7096,7097,7098,7099,7100,7101,7102,7103,7104,7105,7106,7107,7108,7109,7110,7111,7112,7113,7114,7115,7116,7117,7118,7119,7120,7121,7122,7123,7124,7125,7126,7127,7128,7129,7130,7131,7132,7133,7134,7135,7136,7137,7138,7139,7140,7141,7142,7143,7144,7145,7146,7147,7148,7149,7150,7151,7152,7153,7154,7155,7156,7157,7158,7159,7160,7161,7162,7164,7165,7166,7167,7168,7169,7170,7171,7172,7173,7174'
    ref_com = '3566,3774,3567,3568,3569,3570,3571,3572,3573,3574,3575,3576,3577,3578,3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3590,3591,3592,3593,3594,3595,3596,3597,3598,3599,3600,3601,3602,3603,3604,3605,3606,3607,3608,3609,3610,3611,3612,3613,3614,3615,3616,3617,3618,3619,3620,3621,3622,3623,3624,3625,3626,3627,3628,3629,3630,3631,3632,3633,3634,3635,3636,3637,3638,3639,3640,3641,3642,3643,3644,3645,3646,3647,3648,3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,3667,3668,3669,3670,3671,3672,3673,3674,3675,3676,3677,3678,3679,3680,3681,3682,3683,3684,3685,3686,3687,3688,3689,3690,3691,3692,3693,3694,3695,3696,3697,3698,3699,3700,3701,3702,3703,3704,3705,3706,3707,3708,3709,3710,3711,3712,3713,3714,3715,3716,3717,3718,3719,3720,3721,3722,3723,3724,3725,3726,3727,3728,3729,3730,3731,3732,3733,3734,3735,3736,3737,3738,3739,3740,3741,3742,3743,3744,3745,3746,3747,3748,3749,3750,3751,3752,3753,3754,3755,3756,3757,3758,3759,3760,3761,3762,3763,3764,3765,3766,3767,3768,3769,3770,3771,3772,3773, 7163,6966,6967,6968,6969,6970,6971,6972,6973,6974,6975,6976,6977,6978,6979,6980,6981,6982,6983,6984,6985,6986,6987,6988,6989,6990,6991,6992,6993,6994,6995,6996,6997,6998,6999,7000,7001,7002,7003,7004,7005,7006,7007,7008,7009,7010,7011,7012,7013,7014,7015,7016,7017,7018,7019,7020,7021,7022,7023,7024,7025,7026,7027,7028,7029,7030,7031,7032,7033,7034,7035,7036,7037,7038,7039,7040,7041,7042,7043,7044,7045,7046,7047,7048,7049,7050,7051,7052,7053,7054,7055,7056,7057,7058,7059,7060,7061,7062,7063,7064,7065,7066,7067,7068,7069,7070,7071,7072,7073,7074,7075,7076,7077,7078,7079,7080,7081,7082,7083,7084,7085,7086,7087,7088,7089,7090,7091,7092,7093,7094,7095,7096,7097,7098,7099,7100,7101,7102,7103,7104,7105,7106,7107,7108,7109,7110,7111,7112,7113,7114,7115,7116,7117,7118,7119,7120,7121,7122,7123,7124,7125,7126,7127,7128,7129,7130,7131,7132,7133,7134,7135,7136,7137,7138,7139,7140,7141,7142,7143,7144,7145,7146,7147,7148,7149,7150,7151,7152,7153,7154,7155,7156,7157,7158,7159,7160,7161,7162,7164,7165,7166,7167,7168,7169,7170,7171,7172,7173,7174'
    staples = '11735,11736,11737,11738,11739,11740,11741,11742,11743,11744,11745,11746,11747,11748,11749,11750,11751,11752,11753,11754,11755,11756,11757,11758,11759,11760,11761,11762,11763,11764,11765,11766,12249,12250,12251,12252,12253,12254,12255,12256,12257,12258,12259,12260,12261,12262,12263,12264,12265,12266,12267,12268,12269,12270,12271,12272,12273,12274,12275,12276,12277,12278,12635,12640,12641,12642,12643,12644,12645,12646,12647,12648,12649,12650,12651,12652,12653,12654,12655,12656,12657,12658,12659,12660,12661,12662,12663,12664,12665,12666,12667,12668,12669,12670,12671,12672,12690,12691,12692,12693,12694,12695,12696,12697,12698,12699,12700,12701,12702,12703,12704,12705,12706,12707,12708,12709,12710,12711,12712,12713,12714,12715,12716,12717,12718,12719,12720,13162,13163,13164,13165,13166,13167,13168,13169,13170,13171,13172,13173,13174,13175,13176,13177,13178,13179,13180,13181,13182,13183,13184,13185,13186,13187,13188,13189,13190,13191,13192,13561,13562,13563,13564,13565,13566,13567,13568,13569,13570,13571,13572,13573,13574,13575,13576,13577,13578,13579,13580,13581,13582,13583,13584,13585,13586,13587,13588,13589,13590,13591,13592,14075,14076,14077,14078,14079,14080,14081,14082,14083,14084,14085,14086,14087,14088,14089,14090,14091,14092,14093,14094,14095,14096,14097,14098,14099,14100,14101,14102,14103,14104'
    overhangs = '11718,11719,11720,11721,11722,11723,11724,11725,11726,11727,11728,11729,11730,11731,11732,11733,11734,12232,12233,12234,12235,12236,12237,12238,12239,12240,12241,12242,12243,12244,12245,12246,12247,12248,12623,12624,12625,12626,12627,12628,12629,12630,12631,12632,12633,12634,12636,12637,12638,12639,12673,12674,12675,12676,12677,12678,12679,12680,12681,12682,12683,12684,12685,12686,12687,12688,12689,13145,13146,13147,13148,13149,13150,13151,13152,13153,13154,13155,13156,13157,13158,13159,13160,13161,13544,13545,13546,13547,13548,13549,13550,13551,13552,13553,13554,13555,13556,13557,13558,13559,13560,14058,14059,14060,14061,14062,14063,14064,14065,14066,14067,14068,14069,14070,14071,14072,14073,14074'
    
    # staples = '12690,12691,12692,12693,12694,12695,12696,12697,12698,12699,12700,12701,12702,12703,12704,12705,12706,12707,12708,12709,12710,12711,12712,12713,12714,12715,12716,12717,12718,12719,12720,12641,12642,12643,12644,12645,12646,12647,12648,12649,12650,12651,12652,12653,12654,12655,12656,12657,12658,12659,12660,12661,12662,12663,12664,12665,12666,12667,12668,12669,12670,12671,12672,12640'
    # overhangs = '12673,12688,12674,12675,12676,12677,12678,12679,12680,12681,12682,12683,12684,12685,12686,12687,13551,13552,13553,13554,13549,13548,13555,13547,13546,13556,13557,13558,13545,13559,13544,13560,13550,12631,12630,12629,12628,12632,12633,12627,12634,12636,12635,12637,12638,12639,12626,12625,12624,12623'
    
    plane_a = '12249,12250,12251,12252,12253,12254,12255,12256,12257,12258,12259,12260,12261,12262,12263,12264,12265,12266,12267,12268,12269,12270,12271,12272,12273,12274,12275,12276,12277,12278'
    plane_b = '14076,14077,14078,14079,14080,14081,14082,14083,14084,14085,14086,14087,14088,14089,14090,14091,14092,14093,14094,14095,14096,14097,14098,14099,14100,14101,14102,14103,14104'
    plance_c = '12641,12642,12643,12644,12645,12646,12647,12648,12649,12650,12651,12652,12653,12654,12655,12656,12657,12658,12659,12660,12661,12662,12663,12664,12665,12666,12667,12668,12669,12670,12671,12672,12640'
    
    
    point_a = '2043,2058,2044,2045,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,2057'
    point_b = '1409,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408'
    point_c = com_list + ',' + ref_list
    point_p = '12689,12673,12674,12675,12676,12677,12678,12679,12680,12681,12682,12683,12684,12685,12686,12687,12688'
    
    point_alpha = '5931,5930,5929,5928,5927,5926,5925,5922,5921,5920,5919,5667,5666,5665,5664,5663,5662,5661,5660,5659,5658,5657,5656,5655,5654,5653,5652,5651,5650,5649,5648,5647,5646,5645,5644,5643,5642,5641,5640,5639,5638,5637,5636,5635,5386,5385,5384,5383,5382,5381,5380,5379,5378,5377,5376,5375,5374,5373,5372,5371,5370,5369,5368,5367,5366,5365,5364,5363,5362,5361,5360,5359,5358,5357,5356,5355,5354,5353,5108,5107,5106,5105,5104,5103,5102,5101,5100,5099,5098,5097,5096,5095,5094,5093,5092,5091,5090,5089,5088,5087,5086,5085,5084,5083,5082,5081,5080,5079,5078,5077,5076,5075,5074,5073,5072,5071,4824,4823,4822,4821,4820,4816,4815,4814,4813,4812,4811,2380,2379,2378,2086,2085,2084,2083,2082,2081,2080,2079,2078,2077,2076,2075,2074,2073,2072,2071,2070,2069,2068,2067,2066,2065,2064,2063,2062,2061,2060,2059,2058,2057,2056,2055,2054,2053,2052,2051,2050,2049,2048,2047,2046,2045,2044,2043,2042,2041,2040,2039,2038,2037,2036,2035,2034,2033,2032,1763,1762,1761,1760,1759,1758,1757,1756,1755,1754,1753,1752,1751,1750,1749,1748,1747,1746,1745,1744,1743,1742,1741,1740,1739,1738,1737,1736,1735,1734,1733,1732,1731,1730,1729,1728,1727,1726,1725,1724,1723,1722,1721,1720,1719,1718,1717,1716,1715,1714,1713,1712,1711,1710,1709,1708,1707,1706,1705,1704,1703,1438,1437,1436,1435,1434,1433,1432,1431,1430,1429,1428,1427,1426,1425,1424,1423,1422,1421,1420,1419,1418,1417,1416,1415,1414,1413,1412,1411,1410,1409,1408,1407,1406,1405,1404,1403,1402,1401,1400,1399,1398,1397,1396,1395,1394,1393,1392,1391,1390,1389,1388,1387,1386,1385,1384,1383,1382,1381,1380,1116,1115,1114,1113,1112,1111,1110,1109,1108,1104,1103,1102,1101,1100,1099,1098,1097,1096,1094,1093,1092,1091,1090,1089,1088,1087,1086,10383,10382,10840,10839,10838,10837,10831,10830,10829,10828,10827,10826,10825,10863,10862,10861,10860,10859,10858,10857,10856,10855,10854,10853,10852,10851,10850,10849,10848,10847,11298,11297,11296,11295,11294,11293,11366,11365,11364,11363,11362,11361,11766,11765,11764,11763,11762,11761,11760,11759,11758,11757,11756,11755,11754,11753,11752,11751,11750,11749,11748,11747,11746,11745,11744,11743,11742,11741,11740,11739,11738,11737,11736,11735,11734,11733,11732,11731,11730,11729,11728,11727,11726,11725,11724,11723,11722,11721,11720,11719,11718,11815,11814,11813,11812,11811,11810,11809,11808,11807,11806,11805,11804,11803,11802,11801,11800,11799,11798,11797,11796,11795,11794,11793,11792,11791,11790,11789,11788,11787,11786,11785,11784,11783,11782,11781,11780,11779,11778,11777,11776,11775,11774,11773,11772,11771,11770,11769,11768,11767,12211,12210,12196,12278,12277,12276,12275,12274,12273,12272,12266,12265,12264,12263,12262,12261,12260,12259,12258,12257,12256,12255,12254,12253,12252,12251,12250,12249,12248,12247,12246,12245,12244,12243,12242,12241,12240,12239,12238,12237,12236,12235,12234,12233,12232,12672,12671,12670,12669,12668,12667,12666,12665,12664,12663,12662,12661,12660,12659,12658,12657,12656,12655,12654,12653,12652,12651,12650,12649,12648,12647,12646,12645,12644,12643,12642,12641,12640,12639,12638,12637,12636,12635,12634,12633,12632,12631,12630,12629,12628,12627,12626,12625,12624,12623,12720,12719,12718,12717,12716,12715,12714,12713,12712,12711,12710,12709,12708,12707,12706,12705,12704,12703,12702,12701,12700,12699,12698,12697,12696,12695,12694,12693,12692,12691,12690,12689,12688,12687,12686,12685,12684,12683,12682,12681,12680,12679,12678,12677,12676,12675,12674,12673,13192,13191,13190,13189,13188,13187,13186,13185,13184,13183,13182,13181,13180,13179,13178,13177,13176,13175,13174,13173,13172,13171,13170,13169,13168,13167,13166,13165,13164,13163,13162,13161,13160,13159,13158,13157,13156,13154,13153,13152,13151,13150,13149,13148,13147,13146,13145,13592,13591,13590,13589,13588,13587,13586,13585,13584,13583,13582,13581,13580,13579,13578,13577,13576,13575,13574,13573,13572,13571,13570,13569,13568,13567,13566,13565,13564,13563,13562,13561,13560,13559,13558,13557,13556,13555,13554,13553,13552,13551,13550,13549,13548,13547,13546,13545,13544,13641,13640,13639,13638,13637,13636,13635,13634,13633,13632,13631,13630,13629,13628,13627,13626,13625,13624,13623,13622,13621,13620,13619,13618,13617,13616,13615,13614,13613,13612,13611,13610,13609,13608,13607,13606,13605,13604,13599,14104,14103,14102,14101,14100,14099,14098,14097,14096,14095,14094,14093,14092,14091,14090,14089,14088,14087,14086,14085,14084,14083,14082,14081,14080,14079,14078,14077,14076,14075,14074,14073,14072,14071,14070,14069,14066,14065,14064,14063,14062,14061,14060,14059,14058,14472,14471,14470,14469,14468,14467,14466,14465,14463,14462,14461,14460,14459,14544,14543,14542,14541,14540,14539,14538,14537,14536,14534,14533,14532,14531,14530,14529,14528,14527,14526,14508,14507,14506,14505,14504,14503,14502,14501,14500,14995,14994,14993,14992,14991,14990,14989,14988,14987,14986,14985,14977,14976'
    
    # point_check = '
    particle_indexes = [com_list, ref_list, staples, overhangs,
                        point_a, point_b, point_c, point_p,
                        plane_a, plane_b, plance_c, ref_com,
                        point_alpha
                       ]            
    
    # particle_indexes = [com_list, ref_list, staples, overhangs, ref_com, plane_a, plane_b, plance_c]            
    cms_observable = [{'idx':particle_indexes, 'name':'cms_positions.txt', 'print_every':int(5e3)}]
    
    stiff = 0.2
    xmin = 0
    xmax = 72.787 
    n_windows = 46
    
    equlibration_parameters = {'steps':'5e6', 'print_energy_every': '5e6', 'print_conf_interval':'5e6'}
    production_parameters = {'steps':'1e7', 'print_energy_every': '1e7', 'print_conf_interval':'1e7'}
    
    us_list = [CustomObservableUmbrellaSampling(file_dir, sys) for file_dir, sys in zip(file_dirs,systems)]
    
    
    simulation_manager = SimulationManager()
    
    
    for us in us_list:
        us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff,
                               xmin, xmax, equlibration_parameters, cms_observable,
                               observable=True, print_every=5e3, continue_run=False)
    simulation_manager.worker_manager()
    for us in us_list:
        us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff,
                             xmin, xmax, production_parameters,cms_observable,
                             observable=True, print_every=5e3, name='com_distance.txt',
                             continue_run=False)
    simulation_manager.worker_manager()


if __name__ == '__main__':
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    print(f'Umbrella run time: {toc - tic}')
