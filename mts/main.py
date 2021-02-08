#!/usr/bin/env python3

import algorithms

import argparse
import functools
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import random
import sys
import time
import tqdm 

from multiprocessing import Pool
from timeit import default_timer as timer

dupa = 0

# OPT must be the first algorithm in this list
ALGORITHMS_ONLINE = (
  algorithms.OPT,
  algorithms.LRU,
  # algorithms.OPTMarking,
  algorithms.Marker,
  algorithms.Rand,
  # algorithms.Combrand_lambda((algorithms.Marker,algorithms.LRU)),
)

ALGORITHMS_PRED_NEXT = (
  algorithms.FollowPred,
  # algorithms.FollowPredMarking,
  algorithms.LV_PredMarker, # TODO: Tune parameter gamma
  # algorithms.Combrand_lambda((algorithms.Marker,algorithms.FollowPred)),
  algorithms.Rohatgi_LMarker,
  # algorithms.LNonMarker(True, 0.1),
  algorithms.LNonMarker(True, 0.01),
  # algorithms.LNonMarker(True, 0.001),
  # algorithms.LNonMarker(False, 0.1),
  algorithms.LNonMarker(False, 0.01),
  # algorithms.LNonMarker(False, 0.001),

  # algorithms.BlindOracle(True, 0.1),
  algorithms.BlindOracle(True, 0.01),
  # algorithms.BlindOracle(True, 0.001),
  # algorithms.BlindOracle(False, 0.1),
  algorithms.BlindOracle(False, 0.01),
  # algorithms.BlindOracle(False, 0.001),
)

ALGORITHMS_PRED_CACHE = (
  algorithms.ACEPS_TrustDoubt,
  # algorithms.RobustFTP(True, 0.1),
  algorithms.RobustFTP(True, 0.01),
  # algorithms.RobustFTP(True, 0.001),
  # algorithms.RobustFTP(False, 0.1),
  algorithms.RobustFTP(False, 0.01),
  # algorithms.RobustFTP(False, 0.001),
)

@functools.lru_cache()
def LoadObj(filepath):
  with open(filepath, 'rb') as f:
    return pickle.load(f)

def LoadRequests(filepath, dataset):
  parsed_dict = LoadObj(filepath)
  requests = []
  set_id = list(parsed_dict.keys())[dataset]
  # skip last request because we don't have prediction for it
  for dict_line in parsed_dict[set_id][:-1]:
    requests.append(dict_line['address'])
  return tuple(requests)

def LoadParrotReuseDist(filepath, dataset):
  global dupa
  parsed_dict = LoadObj(filepath)
  dataset_records = parsed_dict[list(parsed_dict.keys())[dataset]]
  reuse_distances = []
  # print(123, len(dataset_records))
  for i in range(0, len(dataset_records)-1):
    request_address = dataset_records[i]['address']
    reuse_dist = -1
    next_cache_addresses = dataset_records[i+1]['cache_lines_address']
    next_cache_reuse_distances = dataset_records[i+1]['cache_lines_reuse_distance']
    print(123, dataset_records[i+1])
    print(next_cache_addresses, request_address)
    for j in range(0, len(next_cache_addresses)):
      if str(next_cache_addresses[j]) == str(request_address):
        reuse_dist = next_cache_reuse_distances[j]
        print('ok')
    # print(i)
    # if reuse_dist != -1:
    #   print('')
    # else:
    #   print('ok')
    # assert(reuse_dist != -1)
    reuse_distances.append(pow(2.0, reuse_dist))
  return tuple(reuse_distances)

def LoadParrotCachePreds(filepath, dataset, k):
  parsed_dict = LoadObj(filepath)
  history = []
  set_id = list(parsed_dict.keys())[dataset]
  for j, dict_line in enumerate(parsed_dict[set_id]):
    cache_line = dict_line['cache_lines_address']
    cache = cache_line + ([None] * (k - len(cache_line)))
    history.append(tuple(cache))
  return history

LABELS = [algorithm.__name__ for algorithm in ALGORITHMS_ONLINE]

PREDICTORS_NEXT = (
  # algorithms.PredLRU,  # LRU predictions
  # algorithms.PredPLECO_BK,  # PLECO predictions, designed for the BK dataset
  # algorithms.PredPopularity,  # simple ad-hoc POPU predictions
  algorithms.PredParrot,
)
# LABELS += [algorithm.__name__ + '+LRU' for algorithm in ALGORITHMS_PRED_NEXT]
# LABELS += [algorithm.__name__ + '+PLECO' for algorithm in ALGORITHMS_PRED_NEXT]
# LABELS += [algorithm.__name__ + '+Popu' for algorithm in ALGORITHMS_PRED_NEXT]
LABELS += [algorithm.__name__ + '+ParrotNext' for algorithm in ALGORITHMS_PRED_NEXT]

PREDICTORS_CACHE = (
  # lambda requests: algorithms.FollowPred(requests, k, algorithms.PredLRU(requests)),
  # lambda requests: algorithms.FollowPred(requests, k, algorithms.PredPLECO_BK(requests)),
  # lambda requests: algorithms.FollowPred(requests, k, algorithms.PredPopularity(requests)),
  # lambda requests: algorithms.FollowPred(requests, k, algorithms.PredParrot(requests)),
  algorithms.PredParrot,
)

LABELS += ['ParrotCache']
# LABELS += [algorithm.__name__ + '+LRU' for algorithm in ALGORITHMS_PRED_CACHE]
# LABELS += [algorithm.__name__ + '+PLECO' for algorithm in ALGORITHMS_PRED_CACHE]
# LABELS += [algorithm.__name__ + '+Popu' for algorithm in ALGORITHMS_PRED_CACHE]
LABELS += [algorithm.__name__ + '+ParrotCache' for algorithm in ALGORITHMS_PRED_CACHE]

def SingleRunOfTryAllAlgorithmsAndPredictors(k, filepath, run, dataset):
  requests = LoadRequests(filepath, dataset)
  assert(len(requests) > 0)
  #total_requests += len(requests)
  #print(f'requests: {len(requests)}, dataset: {dataset}')
  costs = []
  times = []
  parrot_cache = LoadParrotCachePreds(filepath, dataset, k)

  for algorithm in ALGORITHMS_ONLINE:
    start = timer()
    output = algorithm(requests, k)
    times.append(timer() - start)
    algorithms.VerifyOutput(requests, k, output)
    costs.append(algorithms.Cost(output))

  for predictor in PREDICTORS_NEXT:
    if predictor == algorithms.PredParrot:
      reuse_dists = LoadParrotReuseDist(filepath, dataset)
      assert(len(reuse_dists) == len(requests))
      predictions = algorithms.PredParrot(reuse_dists)
    else:
      predictions = predictor(requests)
    for algorithm in ALGORITHMS_PRED_NEXT:
      start = timer()
      output = algorithm(requests, k, predictions)
      times.append(timer() - start)
      algorithms.VerifyOutput(requests, k, output)
      costs.append(algorithms.Cost(output))

  costs.append(algorithms.Cost(parrot_cache))

  for predictor in PREDICTORS_CACHE:
    if predictor == algorithms.PredParrot:
      predictions = parrot_cache
    else:
      predictions = predictor(requests)
    for algorithm in ALGORITHMS_PRED_CACHE:
      start = timer()
      output = algorithm(requests, k, predictions)
      times.append(timer() - start)
      algorithms.VerifyOutput(requests, k, output)
      costs.append(algorithms.Cost(output))

  # if (costs[0] <= 0): # skip if OPT has no miss, ensures to never divide by zero
  #   continue
  return run, costs, times

def TryAllAlgorithmsAndPredictors(k, filepath, datasets, num_runs=1):
  total_costs = [[0 for _ in LABELS] for _ in range(num_runs)]
  total_times = [0 for _ in LABELS]
  # start_aux = timer()

  print(f'filepath: {filepath}')
  total_requests = 0
  for dataset in range(datasets):
    requests = LoadRequests(filepath, dataset)
    assert(len(requests) > 0)
    total_requests += len(requests)

  # for run, dataset in tqdm.tqdm(list(itertools.product(range(num_runs), range(datasets)))):

  with Pool() as pool:
     for run, costs, times in pool.starmap(
        SingleRunOfTryAllAlgorithmsAndPredictors, list(itertools.product((k,), (filepath,), range(num_runs), range(datasets)))):
      for i, cost in enumerate(costs):
        # uncomment the line below if you want to compute the average competitive
        # ratio across instances instead of the total competitive ratio
        # cost /= costs[0]
        total_costs[run][i] += cost
      for i, time in enumerate(times):
        total_times[i] += time

      # print(dataset, ', '.join(
      #   '%s: %0.3f' % (label, cost / costs[0])
      #   for label, cost in zip(LABELS, costs)))


  total_competitive_ratios = np.array(total_costs)
  print(total_competitive_ratios[0])
  print(total_costs[0][0])
  total_competitive_ratios = total_competitive_ratios / total_competitive_ratios[:,0].reshape(-1, 1)
  print()
  print('Maximum std dev among runs: %0.5f' % np.max(np.std(total_competitive_ratios, axis=0)))
  total_competitive_ratios = np.mean(total_competitive_ratios, axis=0)
  print('Total competitive ratio (k=%d):' % k)
  MAX_LABEL_LEN = max(len(label) for label in LABELS)
  for label, cr, tc in zip(LABELS, total_competitive_ratios, total_costs[0]):
    print(('%' + str(MAX_LABEL_LEN) + 's: %0.3f %0.3f') % (label, cr, (total_requests - tc) * 1.0/total_requests))
#for label, c in zip(LABELS, total_costs[0]):
#    print(('%' + str(MAX_LABEL_LEN) + 's: %d') % (label, c))
  print()
  print('Total time:')
  for label, time in zip(LABELS, total_times):
    print(('%' + str(MAX_LABEL_LEN) + 's: %0.2fs') % (label, time))
  # aux_time = timer() - start_aux - sum(total_times)
  # print('Aux time: %0.2fs' % aux_time)

  print()
  print('dataset,' + ','.join(LABELS))
  print(filepath + ',' + ','.join(map(str,total_competitive_ratios)))
  total_hit_rates = (total_requests - np.array(total_costs)) * (1.0 / total_requests)
  total_hit_rates = np.mean(total_hit_rates, axis=0)
  print(filepath + ',' + ','.join(map(str,total_hit_rates)))

  return total_costs


def PlotPredRandom(k, datasets, num_runs, output_basename=None, load_json=None, style='paper'):
  ALGORITHMS = ALGORITHMS_ONLINE + ALGORITHMS_PRED_NEXT + ALGORITHMS_PRED_CACHE
  SIGMAS = [0, 2, 5, 10, 20, 50, 100, 200]
  
  if load_json:
    with open(load_json) as f:
      costs = np.array(json.load(f))
  else:
    costs = np.zeros((len(ALGORITHMS), len(SIGMAS), num_runs))  
    for run, dataset in tqdm.tqdm(list(itertools.product(range(num_runs), datasets))):
      with open(dataset) as f:
        requests = tuple(f)
      assert(len(requests) > 0)

      for i, algorithm in enumerate(ALGORITHMS_ONLINE):
        output = algorithm(requests, k)
        algorithms.VerifyOutput(requests, k, output)
        cost = algorithms.Cost(output)
        for j in range(len(SIGMAS)):
          costs[i][j][run] += cost 

      for j, sigma in enumerate(SIGMAS):
        predictions = algorithms.PredRandom(requests, sigma)
        for i, algorithm in enumerate(ALGORITHMS_PRED_NEXT):
          output = algorithm(requests, k, predictions)
          algorithms.VerifyOutput(requests, k, output)
          costs[len(ALGORITHMS_ONLINE) + i][j][run] += algorithms.Cost(output)
        
        cache_predictions = algorithms.FollowPred(requests, k, predictions)
        algorithms.VerifyOutput(requests, k, cache_predictions)
        for i, algorithm in enumerate(ALGORITHMS_PRED_CACHE):
          output = algorithm(requests, k, cache_predictions)
          algorithms.VerifyOutput(requests, k, output)
          costs[len(ALGORITHMS_ONLINE) + len(ALGORITHMS_PRED_NEXT) + i][j][run] += algorithms.Cost(output)

  if output_basename:
    with open(output_basename + '.json', 'w') as f:
      json.dump(costs.tolist(), f)

  competitive_ratios = costs / costs[0]
  print('Maximum std dev among runs: %0.5f' % np.max(np.std(competitive_ratios, axis=2)))
  competitive_ratios = np.mean(competitive_ratios, axis=2)

  LABELS = [algorithm.__name__ for algorithm in ALGORITHMS]  
  KEEP = {
    "LRU" : "LRU",
    "Marker" : "Marker",
    "LazyTrustDoubt" : "Trust\&Doubt",
    "FollowPred" : "FtP",
    "PredMarker" : "L\&V",  # L&V guaranteed algorithm
    "Rnd(Marker,FollowPred)[eps=0.010000]" : "RobustFtP",
  }
  if style in ('appendix', 'slides'):
    KEEP.update({"LMarker" : "LMarker", "LNonMarker" : "LNonMarker"})
  plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 9,
    'pgf.rcfonts': False,
  })
  LINE = iter(itertools.product(['', '.', '+', 'x'], ['-', '--', '-.', ':']))
  
  plots = {}
  for label, ratios in zip(LABELS, competitive_ratios):
    if label not in KEEP:
      continue
    label = KEEP[label]
    mk, ls = next(LINE)
    plots[label] = plt.plot(SIGMAS, ratios, label=label, markersize=5, marker=mk, ls=ls)[0]
  xlabel = plt.xlabel('Noise parameter $\sigma$ of the synthetic predictor')
  ylabel = plt.ylabel('Competitive ratio')
  if style == 'appendix':
    plt.gcf().set_size_inches(w=5.75, h=3)
    plt.gcf().tight_layout()
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    plt.legend(loc='lower right', ncol=4)
  else:
    assert style in ('paper', 'slides')
    plt.gcf().set_size_inches(w=3.25, h=2.5)
    legend = plt.legend(loc='lower right')
  if output_basename:
    if style in ('paper', 'appendix'):
      plt.savefig(output_basename + '.png', dpi=150)
      plt.savefig(output_basename + '.pgf',
        bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
    else:
      assert style == 'slides'
      SLIDES = (
        ('LRU', 'Marker', 'FtP'),
        ('LRU', 'Marker', 'FtP', 'LMarker', 'LNonMarker', 'L\&V'),
        ('LRU', 'Marker', 'FtP', 'L\&V', 'RobustFtP'),
        ('LRU', 'Marker', 'FtP', 'L\&V', 'RobustFtP', 'Trust\&Doubt'),
      )
      for i, slide in enumerate(SLIDES):
        for label, plot in plots.items():
          if label in slide:
            plot.set_visible(True)
            plot.set_label(label)
            plot.set_linewidth(4 if label == slide[-1] else 1)
          else:
            plot.set_visible(False)
            plot.set_label('_' + label)
        plt.legend(loc='lower right')
        plt.savefig(output_basename + str(i) + '.png', dpi=150)
        plt.savefig(output_basename + str(i) + '.pgf',
           bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
  else:
    plt.show()


def PlotEpsilon(k, datasets):
  EPSILONS = [0.5, 0.1, 0.01, 0.001]
  ALGORITHMS_ONLINE = (algorithms.OPT, algorithms.LRU)
  ALGORITHMS_PRED = (
    algorithms.FollowPred,
    algorithms.Combdet_lambda([algorithms.LRU, algorithms.FollowPred]),
  ) + tuple(
    algorithms.Combrand_lambda([algorithms.LRU, algorithms.FollowPred], eps)
    for eps in EPSILONS)
  ALGORITHMS = ALGORITHMS_ONLINE + ALGORITHMS_PRED
  SIGMAS = range(0, 50, 5)
  costs = np.zeros((len(ALGORITHMS), len(SIGMAS)))

  for dataset in tqdm.tqdm(datasets):
    with open(dataset) as f:
      requests = tuple(f)
    for i, algorithm in enumerate(ALGORITHMS_ONLINE):
        output = algorithm(requests, k)
        algorithms.VerifyOutput(requests, k, output)
        cost = algorithms.Cost(output)
        for j in range(len(SIGMAS)):
          costs[i][j] += cost 
    for j, sigma in enumerate(SIGMAS):
      predictions = algorithms.PredRandom(requests, sigma)
      for i, algorithm in enumerate(ALGORITHMS_PRED):
        output = algorithm(requests, k, predictions)
        algorithms.VerifyOutput(requests, k, output)
        costs[len(ALGORITHMS_ONLINE) + i][j] += algorithms.Cost(output)
  competitive_ratios = costs / costs[0]
  for algorithm, ratios in list(zip(ALGORITHMS, competitive_ratios))[1:]:
    plt.plot(SIGMAS, ratios, label=algorithm.__name__)
  plt.legend(loc='lower right');
  plt.show()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', type=str, choices=['all', 'plot', 'eps', 'opt'])
  parser.add_argument('-k', type=int, required=True, help='cache size')
  parser.add_argument('-n', '--num_runs', type=int, default=1, help='number of runs')
  parser.add_argument('-o', '--output_basename', type=str)
  parser.add_argument('-l', '--load_json', type=str)
  parser.add_argument('-s', '--style', type=str, choices=['paper', 'appendix', 'slides'], default='paper')
  # parser.add_argument('DATASETS', type=str, nargs='+')
  parser.add_argument('-d', '--datasets', type=int, default=1, help='number of datasets to process')
  parser.add_argument('-f', '--filename', type=str)
  args = parser.parse_args()

  if args.mode == 'all':
    TryAllAlgorithmsAndPredictors(args.k, args.filename, args.datasets, num_runs=args.num_runs)
  elif args.mode == 'plot':
    PlotPredRandom(args.k, args.DATASETS, args.num_runs, args.output_basename, args.load_json, args.style)
  elif args.mode == 'eps':
    PlotEpsilon(args.k, args.DATASETS)
  elif args.mode == 'opt':
    for dataset in args.DATASETS:
      with open(dataset) as f:
        requests = tuple(f)
      print(algorithms.Cost(algorithms.OPT(requests,args.k)))
  else:
    assert False


if __name__ == '__main__':
  main()
  print(dupa)