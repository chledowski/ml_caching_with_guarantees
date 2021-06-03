import matplotlib.pyplot as plt

MCFBR = [(0, 21), (35, 37), (41, 45), (58, 59), (60, 66), (74, 75), (78, 82), (103, 108), (122, 123), (139, 142), (151, 157), (163, 166), (203, 211), (218, 223), (256, 257), (262, 333), (371, 373), (394, 397), (426, 428), (443, 449), (512, 514), (515, 531), (599, 600), (606, 608), (658, 659), (662, 664), (692, 699), (735, 737), (741, 748), (769, 772), (803, 807), (816, 820), (821, 824), (826, 827), (870, 871), (877, 882), (903, 904), (939, 941), (978, 979), (980, 981), (1027, 1030), (1042, 1046), (1104, 1105), (1113, 1114), (1124, 1126), (1141, 1143), (1169, 1170), (1177, 1179), (1181, 1182), (1187, 1191), (1193, 1197), (1210, 1217), (1234, 1235), (1264, 1265), (1278, 1279), (1293, 1294), (1297, 1298), (1318, 1320), (1344, 1347), (1403, 1405), (1406, 1410), (1418, 1423), (1443, 1456), (1472, 1481), (1507, 1510), (1527, 1528), (1541, 1542), (1553, 1557), (1595, 1597), (1613, 1621), (1622, 1627), (1654, 1659), (1662, 1665), (1687, 1691), (1714, 1716), (1747, 1748), (1773, 1780), (1789, 1792), (1800, 1801), (1827, 1828), (1844, 1847), (1885, 1890), (1913, 1915), (1958, 1966), (1980, 1982), (2018, 2023), (2029, 2031), (2040, 2047), (2073, 2084), (2126, 2128), (2172, 2174), (2181, 2182), (2190, 2191), (2253, 2254), (2293, 2294), (2332, 2334), (2338, 2340), (2354, 2357), (2390, 2392), (2405, 2406), (2411, 2414), (2436, 2437), (2459, 2461), (2489, 2498), (2499, 2500), (2526, 2528), (2534, 2540), (2547, 2550), (2565, 2566), (2567, 2570), (2575, 2576), (2584, 2587), (2611, 2613), (2632, 2634), (2638, 2639), (2644, 2647), (2652, 2653), (2671, 2673), (2685, 2690), (2720, 2722), (2769, 2772), (2773, 2776), (2785, 2789), (2805, 2807), (2816, 2817), (2819, 2822), (2836, 2838), (2853, 2857), (2862, 2868), (2879, 2882), (2903, 2904), (2921, 2922), (2937, 2939), (2950, 2952), (2978, 2979), (2998, 2999), (3012, 3016), (3020, 3021), (3033, 3035), (3045, 3047), (3055, 3056), (3071, 3073), (3086, 3090), (3124, 3128), (3176, 3177), (3204, 3206), (3240, 3244), (3253, 3255), (3272, 3273), (3300, 3304), (3350, 3351), (3358, 3361), (3363, 3374), (3375, 3376), (3381, 3386), (3388, 3390), (3398, 3461), (3480, 3484), (3505, 3511), (3530, 3532), (3537, 3542), (3547, 3549), (3557, 3565), (3570, 3573), (3581, 3582), (3611, 3617), (3637, 3639), (3647, 3648), (3664, 3676), (3693, 3694), (3700, 3701), (3718, 3722), (3753, 3754), (3762, 3763), (3781, 3782), (3793, 3795), (3806, 3810), (3842, 3845), (3866, 3870), (3894, 3897), (3917, 3920), (3921, 3922), (3928, 3931), (3947, 3949), (3957, 3959), (3981, 3983), (4001, 4002), (4022, 4023), (4025, 4027), (4047, 4048), (4075, 4078), (4087, 4089), (4098, 4099), (4101, 4102), (4119, 4121), (4128, 4131), (4145, 4147), (4150, 4151), (4177, 4178), (4185, 4187), (4227, 4230), (4239, 4241), (4267, 4272), (4299, 4301), (4336, 4339), (4379, 4390), (4398, 4399), (4424, 4429), (4443, 4451), (4468, 4469), (4492, 4495), (4526, 4528), (4570, 4573), (4579, 4582), (4596, 4597), (4668, 4674), (4691, 4694), (4721, 4724), (4747, 4748), (4753, 4756), (4783, 4787), (4844, 4845), (4861, 4864), (4866, 4868), (4899, 4901), (4906, 4907), (4912, 4917), (4936, 4939), (4941, 4942), (4955, 4957), (4978, 4982), (5043, 5044), (5062, 5064), (5094, 5099), (5136, 5138), (5187, 5193), (5231, 5240), (5262, 5268), (5312, 5313), (5316, 5321), (5334, 5335), (5365, 5368), (5404, 5405), (5465, 5469), (5478, 5481), (5512, 5515), (5558, 5563), (5573, 5574), (5588, 5592), (5652, 5655), (5665, 5667), (5680, 5702), (5703, 5704), (5715, 5716), (5749, 5751), (5765, 5766), (5784, 5785), (5798, 5801), (5803, 5804), (5851, 5856), (5864, 5870), (5931, 5933), (5935, 5936)]
MCFBD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 16), (5936, None)]
 
MCFRR = [(0, 0), (214, 223), (324, 326), (328, 335), (357, 363), (377, 386), (572, 582), (599, 603), (676, 679), (728, 734), (798, 799), (870, 871), (936, 937), (977, 979), (1051, 1055), (1056, 1059), (1098, 1100), (1139, 1150), (1209, 1218), (1236, 1238), (1240, 1244), (1249, 1252), (1264, 1265), (1285, 1287), (1299, 1302), (1321, 1326), (1327, 1329), (1387, 1395), (1423, 1424), (1435, 1438), (1457, 1462), (1485, 1486), (1504, 1512), (1515, 1519), (1526, 1529), (1572, 1575), (1632, 1634), (1660, 1661), (1693, 1705), (1711, 1714), (1731, 1734), (1781, 1782), (1806, 1812), (1823, 1828), (1840, 1841), (1899, 1903), (1909, 1915), (1931, 1933), (1947, 1952), (2007, 2010), (2017, 2031), (2035, 2039), (2080, 2084), (2095, 2099), (2114, 2115), (2129, 2132), (2144, 2147), (2162, 2165), (2176, 2178), (2221, 2225), (2274, 2277), (2286, 2287), (2334, 2340), (2370, 2373), (2432, 2435), (2436, 2441), (2472, 2498), (2499, 2500), (2503, 2508), (2512, 2516), (2524, 2528), (2552, 2561), (2611, 2613), (2638, 2643), (2660, 2670), (2683, 2690), (2694, 2701), (2721, 2722), (2729, 2732), (2738, 2743), (2762, 2763), (2802, 2804), (2812, 2814), (2849, 2851), (2883, 2886), (2937, 2939), (2942, 2944), (2956, 2959), (2993, 2995), (2996, 2997), (3027, 3028), (3072, 3077), (3084, 3085), (3087, 3090), (3121, 3123), (3124, 3128), (3142, 3144), (3153, 3165), (3174, 3175), (3176, 3177), (3191, 3195), (3213, 3215), (3231, 3234), (3236, 3237), (3239, 3249), (3260, 3267), (3269, 3270), (3297, 3299), (3309, 3312), (3342, 3345), (3350, 3351), (3457, 3458), (3486, 3487), (3500, 3503), (3505, 3511), (3525, 3533), (3550, 3555), (3579, 3580), (3592, 3626), (3637, 3640), (3647, 3650), (3674, 3676), (3707, 3711), (3717, 3724), (3745, 3752), (3754, 3761), (3767, 3769), (3792, 3794), (3808, 3810), (3814, 3817), (3824, 3826), (3833, 3847), (3850, 3855), (3871, 3872), (3916, 3922), (3950, 3954), (4011, 4015), (4117, 4119), (4154, 4161), (4185, 4190), (4191, 4199), (4218, 4220), (4225, 4230), (4239, 4241), (4263, 4264), (4320, 4332), (4355, 4360), (4408, 4414), (4415, 4417), (4422, 4423), (4428, 4430), (4436, 4438), (4448, 4451), (4470, 4473), (4518, 4522), (4529, 4534), (4541, 4542), (4563, 4568), (4579, 4582), (4634, 4640), (4641, 4644), (4645, 4646), (4650, 4651), (4680, 4683), (4684, 4685), (4698, 4702), (4711, 4714), (4738, 4741), (4757, 4765), (4790, 4795), (4801, 4805), (4815, 4816), (4844, 4845), (4854, 4860), (4901, 4904), (4912, 4917), (4920, 4921), (4987, 4988), (4994, 4995), (5012, 5013), (5017, 5020), (5021, 5026), (5043, 5044), (5048, 5051), (5076, 5081), (5088, 5090), (5098, 5099), (5113, 5120), (5121, 5122), (5125, 5128), (5209, 5210), (5221, 5224), (5233, 5240), (5252, 5257), (5268, 5293), (5308, 5311), (5341, 5343), (5365, 5368), (5383, 5385), (5395, 5396), (5418, 5424), (5451, 5458), (5459, 5460), (5645, 5648), (5652, 5661), (5708, 5709), (5727, 5735), (5757, 5767), (5772, 5774), (5831, 5833), (5851, 5857), (5859, 5861), (5862, 5863), (5901, 5904), (5905, 5907), (5931, 5934)]
MCFRD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 16), (5934, None)]

ASTARBR = [(0, 100), (102, 150), (162, 218), (222, 265), (273, 484), (492, 497), (499, 556), (568, 578), (589, 605), (610, 638), (639, 783), (784, 788), (789, 851), (856, 973), (975, 979), (983, 984), (989, 990), (992, 994), (995, 1010), (1027, 1074), (1083, 1093), (1095, 1123), (1130, 1173), (1176, 1180), (1182, 1187), (1188, 1281), (1283, 1285), (1289, 1301), (1323, 1334), (1341, 1366), (1367, 1368), (1373, 1472), (1473, 1482), (1484, 1485), (1488, 1492), (1493, 1557), (1562, 1570), (1572, 1644), (1646, 1662), (1663, 1718), (1721, 1725), (1726, 1734), (1735, 1773), (1775, 1781), (1782, 1797), (1801, 1807), (1825, 1909), (1910, 1915), (1916, 1918), (1923, 1934), (1935, 1985), (1986, 1987), (1989, 1991), (1996, 2046), (2048, 2094), (2096, 2104), (2105, 2106), (2108, 2119), (2123, 2225), (2229, 2231), (2244, 2259), (2261, 2281), (2291, 2326), (2328, 2334), (2337, 2339), (2343, 2379), (2383, 2389), (2396, 2408), (2411, 2472), (2477, 2519), (2521, 2605), (2611, 2651), (2658, 2721), (2731, 2737), (2742, 2757), (2758, 2759), (2764, 2789), (2791, 2809), (2813, 2836)]
ASTARBD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 14), (18, 21), (22, 27), (29, 31), (32, 33), (34, 35), (37, 38), (40, 43), (46, 2836)]

ASTARRR = [(0, 0), (46, 222), (223, 268), (270, 373), (384, 465), (483, 549), (550, 601), (604, 631), (636, 741), (750, 817), (819, 944), (957, 958), (967, 995), (1006, 1093), (1097, 1119), (1121, 1148), (1156, 1201), (1204, 1218), (1228, 1290), (1296, 1325), (1330, 1345), (1359, 1425), (1426, 1453), (1467, 1472), (1475, 1478), (1481, 1536), (1542, 1585), (1587, 1636), (1639, 1676), (1684, 1710), (1712, 1773), (1776, 1788), (1792, 1801), (1810, 1878), (1881, 1904), (1905, 1933), (1941, 2078), (2084, 2086), (2091, 2105), (2108, 2145), (2146, 2162), (2163, 2165), (2166, 2169), (2170, 2238), (2240, 2279), (2280, 2335), (2337, 2342), (2345, 2390), (2392, 2451), (2461, 2466), (2467, 2479), (2484, 2532), (2535, 2591), (2592, 2632), (2634, 2695), (2696, 2697), (2700, 2734), (2744, 2762), (2764, 2788), (2789, 2814), (2831, 2836)]
ASTARRD = [(0, 2), (3, 5), (6, 7), (9, 10), (11, 14), (18, 21), (22, 27), (29, 31), (32, 33), (34, 35), (37, 38), (40, 43), (46, 2836)]

def plot(ax, data):
  for x, y in data:
    if y is None:
      continue
    ax.plot([x, y], [0, 0], '.-', color='C0', markersize=1)
  for (x1, y1), (x2, y2) in zip(data, data[1:]):
    ax.plot([y1, x2], [1, 1], '.-', color='C0', markersize=1)

def plotdataset(BR, BD, RR, RD, title, fname):
  fig, axs = plt.subplots(4)
  plt.setp(axs, yticks=[0, 1], yticklabels=['Predictor', 'Marker'], ylim=[-0.4,1.4])
  plot(axs[0], BR)
  plot(axs[1], BD)
  plot(axs[2], RR)
  plot(axs[3], RD)
  axs[0].set_title('randomized BlindOracle')
  axs[1].set_title('determinstic BlindOracle')
  axs[2].set_title('randomized RobustFtP')
  axs[3].set_title('determinstic RobustFtP')
  plt.xlabel('Time')
  plt.suptitle(title)
  # plt.tight_layout()
  plt.subplots_adjust(hspace=1.5)
  # plt.gcf().set_size_inches(w=1.75*3.5, h=1.75*4.8)
  plt.gcf().set_size_inches(w=1.67*3.5, h=1.67*2.75)
  plt.savefig(fname)  

plotdataset(ASTARBR, ASTARBD, ASTARRR, ASTARRD,
  "astar 100%, predictors are better than Marker",
  'figure-switching-astar.pdf')

plotdataset(MCFBR, MCFBD, MCFRR, MCFRD,
  "mcf 0.01%, Marker is better than predictors",
  'figure-switching-mcf.pdf')

# fig, axs = plt.subplots(8)
# # plt.rcParams.update({'axes.titlesize': 'large', 'font.family': 'serif',  'pgf.rcfonts': False,  })
# # plt.rcParams.update({'font.size': 10,})

# plt.setp(axs, yticks=[0, 1], yticklabels=['Predictor', 'Marker'], ylim=[-0.4,1.4])

# plot(axs[0], ASTARBR)
# plot(axs[1], ASTARBD)
# plot(axs[2], ASTARRR)
# plot(axs[3], ASTARRD)
# plot(axs[4],   MCFBR)
# plot(axs[5],   MCFBD)
# plot(axs[6],   MCFRR)
# plot(axs[7],   MCFRD)

# axs[0].set_title('astar 100% (Predictor is better), randomized BlindOracle')
# axs[1].set_title('astar 100% (Predictor is better), determinstic BlindOracle')
# axs[2].set_title('astar 100% (Predictor is better), randomized RobustFtP')
# axs[3].set_title('astar 100% (Predictor is better), determinstic RobustFtP')
# axs[4].set_title('mcf 0.01% (Marker is better), randomized BlindOracle')
# axs[5].set_title('mcf 0.01% (Marker is better), determinstic BlindOracle')
# axs[6].set_title('mcf 0.01% (Marker is better), randomized RobustFtP')
# axs[7].set_title('mcf 0.01% (Marker is better), determinstic RobustFtP')

# # axs[7].set_xlabel('Time')
# plt.xlabel('Time')

# # plt.tight_layout()
# plt.subplots_adjust(hspace=1.5)

# # plt.gcf().set_size_inches(w=1.75*3.5, h=1.75*4.8)
# plt.gcf().set_size_inches(w=1.75*3.5, h=1.75*4.8)

# plt.savefig('figure-switching.pdf')
