#!/usr/bin/env python

import time
import math
import numpy as np
import random
import scipy.sparse.linalg

class Base(object):
  """
  This class is a base class with common methods.
  """

  def __str__(self):
    """
    Returns this object's string representation.
    """

    if hasattr(self, '__unicode__'):
      return unicode(self).encode('utf-8')
    return "%s object" % self.__class__.__name__

  def __repr__(self):
    """
    Returns this object's string representation.
    """

    return self.__str__()

class Person(Base):
  """
  This class represents a person.
  """

  _last_id = 0

  def __init__(self, name=None):
    """
    Class constructor.

    @param name: Name of a person.
    """

    if not name:
      name = u"P%d" % Person._last_id
      Person._last_id += 1
    self.name = name
    # Default is to not have delegates.
    self._delegates = []

  def __unicode__(self):
    """
    Returns this person's unicode representation, their name.
    """

    return u"%s" % self.name

  def __cmp__(self, other):
    """
    Compares this delegate to another.

    Compares by their string representation and then by their identity ("address").
    """

    if unicode(self) != unicode(other):
      return cmp(unicode(self), unicode(other))
    else:
      return cmp(id(self), id(other))

  def __hash__(self):
    """
    Returns hash value for this person.
    """

    return hash(unicode(self)) ^ id(self)

  def delegates(self, delegates=None):
    """
    Gets or sets this person's delegates.

    @param delegates: A list of delegates to set.
    """

    if delegates is not None:
      sum = 0.0
      delegates_dict = {}
      for d in delegates:
        if d.ratio <= 0.0 or d.ratio > 1.0:
          raise ValueError("Invalid ratio: %f" % d.ratio)
        else:
          sum += d.ratio
        if d.person == self:
          raise ValueError("Delegates contain person themselves.")
        if d.person in delegates_dict:
          delegates_dict[d.person].ratio += d.ratio
        else:
          delegates_dict[d.person] = d
      if delegates and sum != 1.0:
        raise ValueError("Sum of all ratios is not 1: %.12f" % sum)
      self._delegates = delegates_dict.values()
      self._delegates.sort(reverse=True)

    return self._delegates

class Delegate(Base):
  """
  This class represents a delegate for a person.
  """

  def __init__(self, person, ratio):
    """
    Class constructor.

    @param person: A person this delegate is.
    @param ratio: A ratio of trust for this delegate. It has to be a number from (0, 1].
    """

    if (ratio <= 0.0 or ratio > 1.0):
      raise ValueError("Invalid ratio: %f" % ratio)

    self.person = person
    self.ratio = ratio

  def __unicode__(self):
    """
    Returns this delegate's unicode representation, them and their ratio.
    """

    return u"%s(%.2f)" % (self.person, self.ratio)

  def __cmp__(self, other):
    """
    Compares this delegate to another.

    Compares first by their ratio and then by them.
    """

    if self.ratio != other.ratio:
      return cmp(self.ratio, other.ratio)
    else:
      return cmp(self.person, other.person)

  def __hash__(self):
    """
    Returns hash value for this delegate.
    """

    return hash(self.ratio) ^ hash(self.person)

class Vote(Base):
  """
  This class represents a vote.
  """

  def __init__(self, person, vote):
    """
    Class constructor.

    @param person: Person this vote is from.
    @param vote: The value of the vote. It has to be a number from [-1, 1].
    """

    if vote < -1 or vote > 1:
      raise ValueError("Invalid vote: %f" % vote)

    self.person = person
    self.vote = vote

    # For debugging.
    self._debug_values = None

  def __unicode__(self):
    """
    Returns this vote's unicode representation.
    """

    return u"%s(%.2f)" % (self.person, self.vote)

  def __cmp__(self, other):
    """
    Compares this vote to another.

    Compares first by their person and then by a vote.
    """

    if self.person != other.person:
      return cmp(self.person, other.person)
    else:
      return cmp(self.vote, other.vote)

class LinearDelegation(object):
  @classmethod
  def _least_squares(cls, a, b):
    return np.linalg.lstsq(a, b)[0]

  @classmethod
  def _solve(cls, a, b):
    return np.linalg.solve(a, b)

  @classmethod
  def compute_all_votes(cls, persons, votes):
    """
    Compute all delegated votes.

    @param persons: A population.
    @param votes: Votes made.

    @return: A list of made votes plus delegated votes.
    """
    
    n = len(persons)

    persons_to_index = {}
    for i, person in enumerate(persons):
      persons_to_index[person] = i

    known_votes = np.zeros((n, 1))
    persons_who_voted = np.zeros((n, 1))

    for vote in votes:
      known_votes[persons_to_index[vote.person], 0] = vote.vote
      persons_who_voted[persons_to_index[vote.person], 0] = 1.0
        
    delegations_only = np.zeros((n, n))

    for person in persons:
      if persons_who_voted[persons_to_index[person], 0] > 0.0:
        continue

      for delegate in person.delegates():
        delegations_only[persons_to_index[person], persons_to_index[delegate.person]] = delegate.ratio

    delegations = np.identity(n) - delegations_only

    # Computing a least-squares solution to work also when there are cycles of non-voting people.
    computed_has_voted = cls._least_squares(delegations, persons_who_voted)

    computed_has_voted[computed_has_voted < 1e-12] = 0.0

    # We set to zero delegations to all people for who we are unable to compute votes.
    for i, voted in enumerate(computed_has_voted):
      if voted == 0:
        # We set the whole column.
        delegations[:, i] = 0.0
        # But keep the (i, i) element set to 1.0.
        delegations[i, i] = 1.0

    # Normalize delegations to account for those delegations we set to zero above.
    for i, delegation in enumerate(delegations):
      sum = delegations[i].sum()
      # If delegations are defined, sum is not 1.0.
      if sum != 1.0:
        # We subtract 1.0 to account for the (i, i) element, and negate.
        delegations[i] = delegations[i] / -(sum - 1.0)
        delegations[i, i] = 1.0

    computed_votes = cls._solve(delegations, known_votes)

    all_votes = []
    for i, vote in enumerate(computed_votes):
      if computed_has_voted[i] != 0:
        all_votes.append(Vote(persons[i], vote))

    if all_votes:
      all_votes[0]._debug_values = {
        'delegations': delegations,
        'known_votes': known_votes,
        'persons_who_voted': persons_who_voted,
        'computed_has_voted': computed_has_voted,
        'computed_votes': computed_votes,
      }

    return all_votes

class SparseLinearDelegation(LinearDelegation):
  @classmethod
  def _least_squares(cls, a, b):
    return scipy.sparse.linalg.lsqr(scipy.sparse.csr_matrix(a), b)[0].reshape((-1, 1))

  @classmethod
  def _solve(cls, a, b):
    return scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(a), b)

def compute_results(votes):
  """
  Compute the result (average of all votes).

  @param votes: Votes made.

  @return: Voting result.
  """

  sum = 0.0
  for v in votes:
    sum += v.vote

  return sum / len(votes)

def checkEqual(iterator):
  try:
     iterator = iter(iterator)
     first = next(iterator)
     return all(abs(first - rest) < 1e-12 for rest in iterator)
  except StopIteration:
     return True

def print_results(persons, votes, results_votes):
  # print u"Delegations:"
  #
  # for p in persons:
  #   print u" %s:" % p.name
  #   for s in p.delegates():
  #     print u"  %s" % s
  #
  # print u"Votes:"
  #
  # for v in votes:
  #   print u" %s" % v

  for i, all_votes in enumerate(results_votes):
    # print u"Delegated votes %s:" % i
	#
    # for v in sorted(all_votes):
    #   print u" %s" % v

    for v in sorted(all_votes):
      if v._debug_values is not None:
        print "---"
        for key, value in v._debug_values.items():
          print key
          print value

def main():
  """
  Main function.
  """

  random_examples()

def small_example():
  size = 7
  persons = [Person() for i in range(size)]
  persons[1].delegates([Delegate(persons[0], 0.75), Delegate(persons[3], 0.25)])
  persons[2].delegates([Delegate(persons[3], 1.0)])
  persons[3].delegates([Delegate(persons[1], 0.40), Delegate(persons[2], 0.40), Delegate(persons[4], 0.20)])
  persons[5].delegates([Delegate(persons[6], 1.00)])
  persons[6].delegates([Delegate(persons[5], 1.00)])

  votes = [Vote(persons[0], 0.7), Vote(persons[2], -0.4)]

  before = time.clock()
  all_votes = LinearDelegation.compute_all_votes(persons, votes)
  after = time.clock()

  result = compute_results(all_votes)

  print u"Result: %.2f, time: %.3fs" % (result, after - before)

  print_results(persons, votes, [all_votes])

def random_examples():
  """
  It makes a random population, a random delegation network with random votes and computes results for all this.
  """

  np.set_printoptions(precision=20)

  for seed in range(1, 5000):
    # We initialize the random generator to a constant so that runs are reproducible.
    random.seed(seed)

    # Size of a random population.
    size = 1000

    # Random population.
    persons = [Person() for i in range(size)]

    # We define random delegates.
    for p in persons:
      sample = random.sample(persons, random.randint(0, min(int(math.sqrt(size)), 100)))
      delegates = [Delegate(s, random.uniform(0.001, 1)) for s in sample if s is not p]
      # If delegates are provided, we repeat multiple times until sum is really 1.0.
      while delegates:
        sum = 0.0
        for s in delegates:
          sum += s.ratio

        if sum == 1.0:
          break

        for s in delegates:
          s.ratio /= sum

        sum = 0.0
        for s in delegates:
          sum += s.ratio

        if sum == 1.0:
          break

        max(delegates, key=lambda d: d.ratio).ratio += 1.0 - sum

        sum = 0.0
        for s in delegates:
          sum += s.ratio

        if sum == 1.0:
          break

        delegates.pop()

      p.delegates(delegates)

    # And some from population randomly vote
    random_sample = random.sample(persons, random.randint(1, size / 2))
    votes = sorted([Vote(p, random.uniform(-1, 1)) for p in random_sample], key=lambda el: el.person)

    results = []
    results_votes = []
    for cls in (LinearDelegation, SparseLinearDelegation):
      before = time.clock()
      all_votes = cls.compute_all_votes(persons, votes)
      after = time.clock()
      results.append(compute_results(all_votes))
      results_votes.append(all_votes)
      print u"Result: %.2f, time: %.3fs" % (results[-1], after - before)

    if not checkEqual(results):
      print results

      print_results(persons, votes, results_votes)

if __name__ == "__main__":
  main()
