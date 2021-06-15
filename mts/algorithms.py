import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
from operator import add

# An algorithm takes as an input a sequence of requests and the cache size k.
# It returns its full history of execution, i.e. a list of length 1+len(requests),
# with the i-th element being a list of length k representing the cache at the
# moment before the i-th request arrives.


# #################### Predictions start here ####################

# Each prediction returns a list of len(requests) integers, which predicts the next time the current request appears

# Synthetic predictions.
# Computes the next occurence of each element, then add a random noise following a lognormal distribution
# Note: the noise adds an offset, so one should only use the relative order of such predictions, not their exact value 
def PredRandom(requests,sigma):
  infinity = len(requests)
  last_time = dict()
  next_occ = [infinity * 10000] * len(requests)
  for t, request in enumerate(requests):
    if request in last_time:
      next_occ[last_time[request] ] = t
    last_time[request] = t

  noise = np.random.lognormal(0,sigma,len(requests))

  return noise + next_occ
  
# emulates LRU predictions
def PredLRU(requests):
  infinity = len(requests)
  pred = []
  for t, request in enumerate(requests):
      pred.append(-t)
  return pred

# Prediction PLECO tailored for the BK dataset (DOI: 10.1145/2566486.2568018)
# Note: expensive computations, quadratic in len(requests)
def PredPLECO_BK(requests):
  infinity = len(requests)
  pred = [] # the predictions we give (1 / probability(being the next element))
  prev_occs = {} # list of previous occurences of each element
  weights = [] # weights[i] = weight of the element requested i steps earlier (weights[0] -> current element)
  sum_weights = 0

  for t, request in enumerate(requests,1): # t starts at 1
    weights.append((t+10)**(-1.8)*math.exp(-t/670)) # weights starts at t=1
    sum_weights += weights[-1]
    if request not in prev_occs:
      prev_occs[request] = []
    prev_occs[request].append(t)
    prob = sum(weights[t-i] for i in prev_occs[request]) / sum_weights # probability that request is the next occurence according to PLECO: t-i in [0;t-1]
    pred.append(1/prob + t-1) # predicted next occurence
  return pred

# Prediction POPU defined in the main paper
def PredPopularity(requests):
  infinity = len(requests)
  pred = []
  count = {} 
  for t, request in enumerate(requests,1): 
    if request not in count:
      count[request] = 0
    count[request] += 1
    pred.append(t + t/count[request]) 
  return pred

def PredParrot(reuse_dists):
  pred = []
  for t, reuse_dist in enumerate(reuse_dists):
    #print(f"{t}: {reuse_dist}")
    pred.append(t + float(reuse_dist))
  return pred
#	return PredRandom(requests, 50.0)

# #################### Predictions end here ####################


# #################### Algorithms start here ####################

# compute the optimal sequence trusting the predictions
# error_probability adds a probability to predict a random eviction instead
def FollowPred(requests, k, pred, error_probability=0.0):
  cache = [None] * k
  next_pred = [math.inf] * k
  history = [tuple(cache),]
  for t, (request,next) in enumerate(zip(requests,pred)):
    if request in cache:
      index_to_evict = cache.index(request)
    elif None in cache:
      index_to_evict = cache.index(None)
    elif request not in cache:
      index_to_evict = next_pred.index(max(next_pred))
      if random.random() < error_probability:
        index_to_evict = random.randint(0, k-1)
    cache[index_to_evict] = request
    next_pred[cache.index(request)] = next
    history.append(tuple(cache))
  # print(history)
  return history

def FollowPredS(requests, k, pred, error_probability=0.0):
  return FollowPred(requests, k, pred, error_probability), None

# compute the optimal solution, by using previous functions
def OPT(requests, k, pred=[]):
  return FollowPred(requests, k, PredRandom(requests,0))


# Marking algorithm that evicts the furthest predicted unmarked element
def FollowPredMarking(requests, k, pred, error_probability=0.0):
  cache = [None] * k
  cache_preds = [-1] * k
  unmarked = list(range(k))
  history = [tuple(cache),]
  phase_elements = []
  for t, request in enumerate(requests):
    if request not in phase_elements:
      phase_elements.append(request)
    if request in cache:
      index = cache.index(request)
    elif None in cache:
      index = cache.index(None)
    else:
      if len(phase_elements) == k+1:
        assert not unmarked, unmarked
        unmarked = list(range(k))
        phase_elements = [request]
      assert unmarked, len(phase_elements)
      index = max((cache_preds[i],i) for i in unmarked)[1]
      if random.random() < error_probability:
        index = random.choice(unmarked)
    cache[index] = request
    if index in unmarked:
      unmarked.remove(index)
    cache_preds[index] = pred[t]
    history.append(tuple(cache))
  return history
  

# optimal marking algorithm
def OPTMarking(requests, k):
  return FollowPredMarking(requests, k, PredRandom(requests,0))


# evicts random elements
def Rand(requests, k, pred=[]):
  cache = [None] * k
  history = [tuple(cache),]
  for t, request in enumerate(requests):
    if request not in cache:
      cache[random.randrange(0,len(cache))] = request
    history.append(tuple(cache))
  return history


# evicts the least recently used element
def LRU(requests, k, pred=[]):
  cache = [None] * k
  time_used = [-1] * k
  history = [tuple(cache),]
  for t, request in enumerate(requests):
    if request not in cache:
      cache[time_used.index(min(time_used))] = request
    time_used[cache.index(request)] = t
    history.append(tuple(cache))
  return history


# log(k)-competitive algorithm
def Marker(requests, k, pred=[]):
  cache = [None] * k
  unmarked = []
  history = [tuple(cache),]
  for t, request in enumerate(requests):
    if request not in cache:
      if not unmarked:
        unmarked = list(range(k))
      cache[random.choice(unmarked)] = request
    index = cache.index(request)
    if index in unmarked:
      unmarked.remove(index)
    history.append(tuple(cache))
  return history


# algorithm from Lykouris & Vassilvitsky (https://dblp.org/rec/conf/icml/LykourisV18.html)
# named L&V in the paper
PRED_MARKER_GAMMA = 1.
def LV_PredMarker(requests, k, pred):
  Hk = 1.
  for i in range(2, k+1):
    Hk += 1/i
  cache = [None] * k
  unmarked = list(range(k))
  cache_preds = [-1] * k
  clean_c = 0
  stale = {}
  chain_lengths = []
  chain_reps = []
  history = [tuple(cache),]
  
  # 0 - removed element with longest reuse distance
  # 1 else
  pred_usage = [0] * 2

  for t, request in enumerate(requests):  
    if request in cache:
      index_to_evict = cache.index(request)
    elif None in cache:
      index_to_evict = cache.index(None) 
    else:
      if not unmarked:
        clean_c = 0
        stale = set(cache)
        unmarked = list(range(k))
        chain_lengths = []
        chain_reps = []
      if request not in stale:
        clean_c += 1
        index_to_evict = max((cache_preds[i],i) for i in unmarked)[1]              
        chain_lengths.append(1)
        chain_reps.append(cache[index_to_evict])
      else: 
        assert request in chain_reps
        c = chain_reps.index(request)
        chain_lengths[c] += 1
        if chain_lengths[c] <= Hk * PRED_MARKER_GAMMA:
          index_to_evict = max((cache_preds[i],i) for i in unmarked)[1]
        else:
          index_to_evict = random.choice(unmarked)
        chain_reps[c] = cache[index_to_evict]      
    
    if cache_preds[index_to_evict] == max(cache_preds):
      pred_usage[0] += 1
    else:
      pred_usage[1] += 1

    cache[index_to_evict] = request
    cache_preds[index_to_evict] = pred[t]
    if index_to_evict in unmarked:
      unmarked.remove(index_to_evict)
    history.append(tuple(cache))
    
  print(f"marker pred %: {ToPercentages(pred_usage)[0]}")
  return history, None # ToPercentages(pred_usage)
      
      
# Algorithm from Rohatgi (https://doi.org/10.1137/1.9781611975994.112)
# weaker competitive ratio than LNonMarker
def Rohatgi_LMarker(requests, k, pred):
  cache = [None] * k
  unmarked = []
  cache_preds = [-1] * k
  stale = {}
  history = [tuple(cache),]

  # 0 - removed element with longest reuse distance
  # 1 else
  pred_usage = [0] * 2

  for t, request in enumerate(requests):
    if request in cache:
      index_to_evict = cache.index(request)
    elif None in cache:
      index_to_evict = cache.index(None)
    else:
      if not unmarked:
        stale = set(cache)
        unmarked = list(range(k))
      if request in stale:
        index_to_evict = random.choice(unmarked)
      else:
        index_to_evict = max((cache_preds[i],i) for i in unmarked)[1]

    if cache_preds[index_to_evict] == max(cache_preds):
      pred_usage[0] += 1
    else:
      pred_usage[1] += 1

    cache[index_to_evict] = request
    cache_preds[index_to_evict] = pred[t]
    if index_to_evict in unmarked:
      unmarked.remove(index_to_evict)
    history.append(tuple(cache))
  print(f"marker pred %: {ToPercentages(pred_usage)[0]}")
  return (history, None) #ToPercentages(pred_usage))


# Algorithm from Rohatgi (https://doi.org/10.1137/1.9781611975994.112)
# Better competitive ratio than LMarker but is not robust: it needs to be combined with Marker
def LNonMarker_nonrobust(requests, k, pred):
  cache = [None] * k
  unmarked = []
  cache_preds = [-1] * k
  stale = {}
  history = [tuple(cache),]
  evictions = {}
  phase_elements = []

  # 0 - removed element with longest reuse distance
  # 1 else
  pred_usage = [0] * 2
  
  for t, request in enumerate(requests):
    if request not in phase_elements:
      phase_elements.append(request)
      
    if len(phase_elements) == k+1:
      stale = set(cache)
      unmarked = list(range(k))
      evictions = {}
      phase_elements = [request]
      
    if request in cache:
      index_to_evict = cache.index(request)
    elif None in cache:
      index_to_evict = cache.index(None)
    else:
      assert unmarked
      if request in stale:
        if evictions[request] not in stale:
          index_to_evict = random.choice(range(k))
        else:
          index_to_evict = random.choice(unmarked)
      else:
        index_to_evict = max((cache_preds[i],i) for i in unmarked)[1]
    
    
    if cache_preds[index_to_evict] == max(cache_preds):
      pred_usage[0] += 1
    else:
      pred_usage[1] += 1
    
    evictions[cache[index_to_evict]] = request
    cache[index_to_evict] = request
    cache_preds[index_to_evict] = pred[t]
    if index_to_evict in unmarked:
      unmarked.remove(index_to_evict)
    history.append(tuple(cache))


  print(f"marker pred %: {ToPercentages(pred_usage)[0]}")
  return history


# Implementation of the new algorithm Trust&Doubt

# pred must be here the output (i.e., history) of a lazy algorithm (i.e., evicting a single element per request)
# This implementation is not lazy, it has to be used as a subroutine in the function LazyTrustDoubt

# different eviction policies
TrustDoubtLRUevict = True
TrustDoubtAncientEvictPredLRU = True

def TrustDoubt(requests, k, pred):
  cache = [None] * k
  priorities = list(range(k)) # priorities[i] is the priority of stale[i]
  history = [tuple(cache),]
  new_elements = [] # elements arrived in this phase
  last_seen = dict()

  for t, request in enumerate(requests):
    last_seen[request] = t
    if request not in new_elements:
      new_elements.append(request)
      
    # start of a new phase
    if len(new_elements) > k or t == 0:  
      ancient = set(cache) - set(new_elements) # cache elements that were not requested this phase, nor in the previous one
      random.shuffle(priorities) # k priorities even is stale can be smaller, this does not matter
      clean_evict = dict() # clean_evict[q] = p_q in the paper, so clean_evict.values() = T | D, where T = {p_q, trusted[q]=true} and D = {p_q, trusted[q] = false}
      trusted = dict()
      clean_pages = [] # C in the paper
      stale = list(set(cache) - set(ancient))
      unmarked_stale = list(stale)
      arrival = dict() # first request time of each element in the current phase
      interval_arrivals = dict() # number of arrivals at the start of the current interval
      interval_length = dict() # length of current interval
      new_elements = [request] # elements arrived in this phase
       
    if request not in arrival:
      interval_arrivals[request] = len(arrival)
      arrival[request] = t
    if request in unmarked_stale:
      unmarked_stale.remove(request)
    
    # in the first part of a phase, we evict ancient elements
    if ancient:
      if request in ancient:
        ancient.remove(request)
      if request not in cache:
        if None in cache:
          cache[cache.index(None)] = request
        else:
          # evict some ancient page, several policies are possible
          if TrustDoubtAncientEvictPredLRU:
            candidates = ancient - set(pred[t+1])
            if not candidates:
              candidates = ancient
          else:
            candidates = ancient
          evict = min((last_seen[x] if x in last_seen else -1, x)  for x in candidates)[1] # LRU rule
          assert evict in cache
          ancient.remove(evict)
          cache[cache.index(evict)] = request    
    else:
      # ancient is now empty
    
      if request not in stale and arrival[request] == t: # clean pages are defined only when ancient is empty. They are non-stale element that arrived
        clean_pages.append(request)

      # step 1: evict the lowest priority
      if request not in cache:
        assert set(unmarked_stale) & set(cache)
        index_to_evict = min((priorities[stale.index(p)], cache.index(p)) for p in unmarked_stale if p in cache)[1]
        cache[index_to_evict] = request

      # step 2: initialize p_q and trusted
      if request in clean_pages and arrival[request] == t:
        # choose a page unmarked stale or arrived at this phase, not in the predictor's cache, and not in T, not in D
        candidates = list(((set(unmarked_stale) | set(new_elements)) - set (pred[t+1])) - set(clean_evict.values()))
        if not candidates:
          assert None in cache or None in history[t-1]
          candidates = [None]
        if TrustDoubtLRUevict:
          clean_evict[request] = min((last_seen[x] if x in last_seen else -1, x)  for x in candidates)[1]
        else:
          clean_evict[request] = random.choice(candidates)
        trusted[request] = True
        interval_length[request] = 1

      # step 3: if we charged the eviction of the current request was evicted by page q, reset the page that q evicted, without trusting the predictor 
      for q, pq in clean_evict.items():
        if request == pq:
          candidates = list(((set(unmarked_stale) | set(new_elements)) - (set (pred[t+1]))) - (set(clean_evict.values())))

          assert candidates
          if TrustDoubtLRUevict:
            clean_evict[q] = min((last_seen[x] if x in last_seen else -1, x)  for x in candidates)[1]
          else:
            clean_evict[q] = random.choice(candidates)
          trusted[q] = False
          break

      # step 4: regularly (i.e., after some number of arrivals), evict p_q and put back in the cache an unmarked page     
      if arrival[request] == t:
        for q in clean_pages:
          if len(arrival) - interval_arrivals[q] == interval_length[q]: 
            interval_arrivals[q] = len(arrival)
            if trusted[q] == False:
              interval_length[q] *= 2
              trusted[q] = True
            assert q in clean_evict, shorten([q])
            if clean_evict[q] in cache and clean_evict[q] != None:
              index = cache.index(clean_evict[q])
              cache[index] = None 
              
              candidates = set(unmarked_stale) - set(cache) # unmarked stale page not in cache
              T = set([ clean_evict[x] for x in trusted if trusted[x] ])
              candidates = candidates - T # keep only pages not in T
                         
              assert candidates
              page = max((priorities[i],p) for i,p in enumerate(stale) if p in candidates)[1] # take the highest priority
              cache[index] = page
    
    assert request in cache
    history.append(tuple(cache))
    
  return history


# Lazy-fication of the above algorithm. This is the algorithm that we consider.
def ACEPS_TrustDoubt(requests, k, pred):
  goal = TrustDoubt(requests, k, pred)
  cur_cache = [None]*k
  history = [tuple(cur_cache),]
  last_used = [-1] * k
  # print(f"req: {len(requests)}")
  # print(f"pred: {len(pred)}")
  follow_pred = [0] * 2 # 0 pred followed, 1 pred not followed
  for t, request in enumerate(requests):
    new_cache = Lazy_update(cur_cache, goal[t+1], request, last_used, t)
    cache_removed = Removed(cur_cache, new_cache)
    pred_removed = Removed(pred[t+1], pred[t])
    if cache_removed is not None:
      if cache_removed in pred[t+1]:
        follow_pred[1] += 1 # removed sth that is in pred cache, so not following pred
      else:
        follow_pred[0] += 1
    elif pred_removed is not None:
      if pred_removed in new_cache:
        follow_pred[1] += 1 # pred removed sth that is in our cache, so we're not following pred here
      else:
        follow_pred[0] += 1
    history.append(new_cache)
    cur_cache = new_cache
  return history, ToPercentages(follow_pred)

# returns element that was removed from cur_cache or None if caches are identical
def Removed(cur_cache, new_cache):
  diff = ((set(cur_cache)-{None})-(set(new_cache)-{None}))
  assert len(diff) <= 1
  if len(diff) == 1:
    return list(diff)[0]
  return None

def ToPercentages(usage):
  s = sum(usage)
  return [x * 100.0 / s for x in usage]

def FollowPredCache(requests, k, pred):
  return pred

# #################### Algorithms end here ####################


# #################### Combining schemes start here ####################

# cost to move from cache1 to cache2
def Cache_cost(cache1, cache2):
  return len((set(cache2)-{None})-(set(cache1)-{None}))
  

# Procedure used to obtain a lazy algorithm (1 eviction / request) from a non-lazy algorithm
# we currently have cache1, we serve the request and aim at simulating cache2
# we evict a single element from cache1 not in cache2 to serve the request at minimal cost while using cache2 as an advice
# we choose the LRU element, and use the parameter last used (last time each element of cache 1 was used)
def Lazy_update(cache1, cache2, request, last_used, t):
  cache = list(cache1)
  if (request in cache):
    return cache
  if None in cache:
    index_to_evict = cache.index(None)
  else:
    candidates =  list(set(cache) - set(cache2))
    assert len(candidates)>0
    if last_used:
      assert len(last_used) == len(cache1)
      index_to_evict = min((last_used[cache1.index(i)],cache1.index(i)) for i in candidates)[1]
      last_used[index_to_evict] = t
    else:
      index_to_evict = cache.index(random.choice(candidates))
  cache[index_to_evict] = request
  return cache

class Stats:
  def __init__(self):
    self.cur_alg = 0
    self.switch_hist = []
    self.pred_use = []
    self.pred_use_start = 0

  def record(self, new_alg, t):
    if new_alg != self.cur_alg:
      self.switch_hist.append(t)
      if self.cur_alg == 0:
        self.pred_use.append(tuple([self.pred_use_start, t-1]))
      else:
        self.pred_use_start = t
      self.cur_alg = new_alg

  def finish_record(self, l):
    if self.cur_alg == 0:
      self.pred_use.append(tuple([self.pred_use_start, l]))
      


# Combine randomly 2 algorithms
# algs: list of algorithms to combine
# parameterized by epsilon
def Combine_rand(requests, k, pred, algs, epsilon=0.01, LAZY=True):
  m = len(algs)
  assert m==2
  histories = list(map(lambda x: x(requests, k, pred), algs))
  weights = [1] * len(algs)
  probs = [1/len(algs)] * len(algs)
  history = [tuple([None] * k),]
  last_used = [-1] * k
  cur_alg = random.randrange(0,len(algs))
  
  switch_count = 0
  usage = [0] * m
  s = Stats()
  for t, request in enumerate(requests):
    loss = list(map(lambda x: Cache_cost(x[t],x[t+1]), histories))
    new_weights = [w*(1-epsilon)**l for w,l in zip(weights,loss)]
    total_weights = sum(new_weights)
    new_probs = [w / total_weights for w in new_weights]


    if (new_probs[cur_alg] < probs[cur_alg]):
      cur_alg =  1-cur_alg if (random.random() < (probs[cur_alg] - new_probs[cur_alg])/probs[cur_alg]) else cur_alg

    s.record(cur_alg, t)
    usage[cur_alg] += 1
    
    probs = new_probs
    weights = new_weights
    weights = probs # prevents floating point errors: always normalize the weights
    
    if (LAZY): # evict a random element which is in our cache but not in the target cache
      history.append(tuple(Lazy_update(history[t], histories[cur_alg][t+1], request, last_used, t)))
    else:
      history.append(histories[cur_alg][t+1])
  
  s.finish_record(len(requests))
  print(f"switch count: {len(s.switch_hist)}")
  # usage = [100.0 * x / len(requests) for x in usage]
  # print(f"pred: {usage[0]}")
  print(f"pred_use: {s.pred_use}")

  fig, ax = plt.subplots()

  x = []
  y = []
  for seg in s.pred_use:
#    ax.plot([seg[0]], [0], 'bo')
    ax.plot(seg, [0, 0], 'b-')
  for (seg1, seg2) in zip(s.pred_use, s.pred_use[1:]):
#    ax.plot([seg1[1]], [1], 'ro')
    ax.plot([seg1[1], seg2[0]], [1, 1], 'r-')
    """if(seg[0] > 0):
      x.append(seg[0] - 1)
      y.append(1)
    x.append(seg[0])
    y.append(0)
    if(seg[1] > seg[0]):
      x.append(seg[1])
      y.append(0)
    x.append(seg[1] + 1)
    y.append(1)"""

  plt.show()
  return history, ToPercentages(usage)


# Combine deterministically 2 algorithms
# algs: list of algorithms to combine
# parameterized by gamma
def Combine_det(requests, k, pred, algs, gamma=1.0, LAZY=True):
  m = len(algs)
  cur_alg = 0
  costs = [0] * m
  bound = 1.
  histories = list(map(lambda x: x(requests, k, pred), algs))
  history = [tuple([None] * k),]
  last_used = [-1] * k
  
  usage = [0] * m
  s = Stats()
  for t, request in enumerate(requests):
    costs = list(map(add, costs , list(map( lambda x: Cache_cost(x[t], x[t+1]), histories)))) # add the new cost for each algorithm
    #old_cur_alg = cur_alg
    while (costs[cur_alg] > bound):
      cur_alg = (cur_alg + 1) % m
      bound *= 1. + gamma / (m - 1)
    usage[cur_alg] += 1
    s.record(cur_alg, t)
    if (LAZY): 
      history.append(tuple(Lazy_update(history[t], histories[cur_alg][t+1], request, last_used, t)))
    else:
      history.append(histories[cur_alg][t+1])
  s.finish_record(len(requests))
  # usage = [100.0 * x / len(requests) for x in usage]
  print(f"switch_count: {len(s.switch_hist)}")
  # print(f"s_switch_hist: {s.switch_hist}")
  print(f"pred_use: {s.pred_use}")

  fig, ax = plt.subplots()

  x = []
  y = []
  for seg in s.pred_use:
#    ax.plot([seg[0]], [0], 'bo')
    ax.plot(seg, [0, 0], 'b-')
  for (seg1, seg2) in zip(s.pred_use, s.pred_use[1:]):
#    ax.plot([seg1[1]], [1], 'ro')
    ax.plot([seg1[1], seg2[0]], [1, 1], 'r-')
    """if(seg[0] > 0):
      x.append(seg[0] - 1)
      y.append(1)
    x.append(seg[0])
    y.append(0)
    if(seg[1] > seg[0]):
      x.append(seg[1])
      y.append(0)
    x.append(seg[1] + 1)
    y.append(1)"""

  plt.show()
  return history, ToPercentages(usage)


### predefined combining schemes

def Combrand_lambda(algs, epsilon=0.01):
  algorithm = lambda requests, k, pred=[]: Combine_rand(requests, k, pred, algs, epsilon)
  algorithm.__name__ = 'Rnd(' + ','.join(alg.__name__ for alg in algs) + ')' + ('[eps=%f]' % epsilon)
  return algorithm


def Combdet_lambda(algs, gamma=0.01):
  algorithm = lambda requests, k, pred=[]: Combine_det(requests, k, pred, algs, gamma)
  algorithm.__name__ = 'Det(' + ','.join(alg.__name__ for alg in algs) + ')' + ('[gamma=%f]' % gamma)
  return algorithm

# Rohatgi's best algorithm
# def LNonMarker(requests, k, pred):
#   return Combine_det(requests, k, pred, (LNonMarker_nonrobust, Marker))
def LNonMarker(randomized=True, parameter=0.01):
  algorithm = (Combrand_lambda if randomized else Combdet_lambda)((LNonMarker_nonrobust, Marker), parameter)
  algorithm.__name__ = 'Rohatgi_LNonMarker%s[%f]' % ('R' if randomized else 'D', parameter)
  return algorithm

def BlindOracle(randomized=True, parameter=0.01):
  algorithm = (Combrand_lambda if randomized else Combdet_lambda)((FollowPred, Marker if randomized else LRU), parameter)
  algorithm.__name__ = 'Wei_BlindOracle%s[%f]' % ('R' if randomized else 'D', parameter)
  return algorithm

def RobustFTP(randomized=True, parameter=0.01):
  algorithm = (Combrand_lambda if randomized else Combdet_lambda)((FollowPredCache, Marker), parameter)
  algorithm.__name__ = 'ACEPS_RobustFollow%s[%f]' % ('R' if randomized else 'D', parameter)
  return algorithm


# #################### Combining schemes end here ####################


# #################### Utilities start here ####################

def VerifyOutput(requests, k, history):
  assert len(history) == len(requests) + 1
  for cache in history:
    assert len(cache) == k
  for element in history[0]:
    assert element is None
  for request, cache in zip(requests, history[1:]):
    assert request in cache


def Cost(history):
  cost = 0
  for cache_prev, cache_next in zip(history[:-1], history[1:]):
    cost += Cache_cost(cache_prev,cache_next) 
  return cost


# #################### Utilities end here ####################
