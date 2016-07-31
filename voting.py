#!/usr/bin/env python

import time
import math
import numpy as np
import random

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
      missing_self = True
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
      if sum > 1.0:
        raise ValueError("Sum of all ratios is larger than 1: %f" % sum)
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

class RecursiveDelegationOne(object):
  @classmethod
  def _delegate_vote(cls, person, votes_dict, pending, visited=[]):
    """
    For persons who did not vote for themselves this function computes a delegate vote.

    @param person: A person to computer delegate vote for.
    @param votes_dict: A dictionary of already computed votes.
    @param pending: A dictionary of pending persons to computer delegate vote for.
    @param visited: A list of delegate votes we have already tried to compute.

    @return: None if the delegate vote for this person is impossible to compute, or a computed vote.
    """

    if person in votes_dict:
      # Vote for this person is already known.
      return votes_dict[person]

    if person not in pending:
      # This person does not exist in current population so we are unable to compute their vote.
      # This probably means that we removed the person from votes_dict and pending as their delegate
      # vote was not computable.
      return None

    if person in visited:
      # In computing the delegate vote we came back to vote we are already computing.
      return None

    delegates = person.delegates()

    votes = []
    for d in delegates:
      assert d.person != person

      votes.append((d.ratio, cls._delegate_vote(d.person, votes_dict, pending, visited + [person])))

    # We remove votes which were not possible to compute.
    known_votes = [(r, v) for (r, v) in votes if v is not None]

    if len(known_votes) == 0:
      # The delegate vote is impossible to compute for this person
      # (we have a subgraph where we cannot do anything).
      return None

    sum = 0.0
    for (r, v) in known_votes:
      sum += r

    # We normalize to votes which were possible to compute.
    known_votes = [(r / sum, v) for (r, v) in known_votes]

    result = 0.0
    for (r, v) in known_votes:
      result += r * v.vote

    # This version computes vote for same person multiple times which makes it inefficient,
    # but also makes values potentially different (if a different set of known votes is used
    # to compute the value during one graph traversal and then later on a different set because
    # some values are already known).
    vote = Vote(person, result)
    vote._debug_values = known_votes

    return vote

  @classmethod
  def compute_all_votes(cls, persons, votes):
    """
    Compute all delegated votes.

    @param persons: A population.
    @param votes: Votes made.

    @return: A list of made votes plus delegated votes.
    """

    if len(votes) == 0:
      raise ValueError("At least one vote has to be cast.")
    if len(persons) == 0:
      raise ValueError("Zero-sized population.")

    # A dictionary of (currently) finalized votes.
    votes_dict = {}

    # Persons we have to calculate delegation for.
    pending = set()

    for v in votes:
      assert v.person not in votes_dict and v.person not in pending
      votes_dict[v.person] = v

    for p in persons:
      if p in votes_dict:
        continue

      assert p not in pending
      pending.add(p)

    # A list of persons we could not compute votes for.
    unable_votes = []

    while len(pending) > 0:
      for p in pending:
        assert p not in votes_dict and p not in unable_votes

        vote = cls._delegate_vote(p, votes_dict, pending)

        if vote is None:
          # This one we will never be able to compute.
          unable_votes.append(p)
        else:
          votes_dict[p] = vote

        pending.remove(p)

        # We just want to pick an arbitrary p from pending in this inner loop.
        # But we cannot continue looping in inner loop because we modified the pending set.
        break

    #if unable_votes:
    #  print "Unable to compute vote for: %s" % ", ".join(["%s" % s for s in sorted(unable_votes)])

    return votes_dict.values()

class RecursiveDelegationTwo(object):
  @classmethod
  def _delegate_vote(cls, person, votes_dict, pending, visited=[]):
    """
    For persons who did not vote for themselves this function computes a delegate vote.

    @param person: A person to computer delegate vote for.
    @param votes_dict: A dictionary of already computed votes.
    @param pending: A dictionary of pending persons to computer delegate vote for.
    @param visited: A list of delegate votes we have already tried to compute.

    @return: None if the delegate vote for this person is impossible to compute, or a computed vote.
    """

    if person in votes_dict:
      # Vote for this person is already known.
      return votes_dict[person]

    if person not in pending:
      # This person does not exist in current population so we are unable to compute their vote.
      # This probably means that we removed the person from votes_dict and pending as their delegate
      # vote was not computable.
      return None

    if person in visited:
      # In computing the delegate vote we came back to vote we are already computing.
      return None

    delegates = person.delegates()

    votes = []
    for d in delegates:
      assert d.person != person

      votes.append((d.ratio, cls._delegate_vote(d.person, votes_dict, pending, visited + [person])))

    # We remove votes which were not possible to compute.
    known_votes = [(r, v) for (r, v) in votes if v is not None]

    if len(known_votes) == 0:
      # The delegate vote is impossible to compute for this person
      # (we have a subgraph where we cannot do anything).
      return None

    sum = 0.0
    for (r, v) in known_votes:
      sum += r

    # We normalize to votes which were possible to compute.
    known_votes = [(r / sum, v) for (r, v) in known_votes]

    result = 0.0
    for (r, v) in known_votes:
      result += r * v.vote

    vote = Vote(person, result)
    vote._debug_values = known_votes

    votes_dict[person] = vote
    pending.remove(person)

    return vote

  @classmethod
  def compute_all_votes(cls, persons, votes):
    """
    Compute all delegated votes.

    @param persons: A population.
    @param votes: Votes made.

    @return: A list of made votes plus delegated votes.
    """

    if len(votes) == 0:
      raise ValueError("At least one vote has to be cast.")
    if len(persons) == 0:
      raise ValueError("Zero-sized population.")

    # A dictionary of (currently) finalized votes.
    votes_dict = {}

    # Persons we have to calculate delegation for.
    pending = set()

    for v in votes:
      assert v.person not in votes_dict and v.person not in pending
      votes_dict[v.person] = v

    for p in persons:
      if p in votes_dict:
        continue

      assert p not in pending
      pending.add(p)

    # A list of persons we could not compute votes for.
    unable_votes = []

    while len(pending) > 0:
      for p in pending:
        assert p not in votes_dict and p not in unable_votes

        vote = cls._delegate_vote(p, votes_dict, pending)

        if vote is None:
          # This one we will never be able to compute.
          unable_votes.append(p)

          # If vote is not None, it has already been removed from pending.
          pending.remove(p)

        # We just want to pick an arbitrary p from pending in this inner loop.
        # But we cannot continue looping in inner loop because we modified the pending set.
        break

    #if unable_votes:
    #  print "Unable to compute vote for: %s" % ", ".join(["%s" % s for s in sorted(unable_votes)])

    return votes_dict.values()

class LinearDelegation(object):
  @classmethod
  def compute_all_votes(cls, persons, votes):
    """
    Compute all delegated votes.

    @param persons: A population.
    @param votes: Votes made.

    @return: A list of made votes plus delegated votes.
    """

    persons_to_index = {}
    for i, person in enumerate(persons):
      persons_to_index[person] = i

    known_votes = np.zeros((len(persons), 1))
    persons_who_voted = np.zeros((len(persons), 1))

    for vote in votes:
      known_votes[persons_to_index[vote.person], 0] = vote.vote
      persons_who_voted[persons_to_index[vote.person], 0] = 1.0

    delegations = np.identity(len(persons))

    for person in persons:
      if persons_who_voted[persons_to_index[person], 0] > 0.0:
        continue

      for delegate in person.delegates():
        delegations[persons_to_index[person], persons_to_index[delegate.person]] = -delegate.ratio

    computed_votes = np.linalg.solve(delegations, known_votes)
    computed_has_voted = np.linalg.solve(delegations, persons_who_voted)

    computed_votes[np.abs(computed_votes) < 1e-12] = 0.0
    computed_has_voted[np.abs(computed_has_voted) < 1e-12] = 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
      normalized_votes = computed_votes / computed_has_voted

    all_votes = []
    it = np.nditer(normalized_votes, flags=['f_index'])
    while not it.finished:
      if np.isfinite(it[0]):
        all_votes.append(Vote(persons[it.index], it[0]))

      it.iternext()

    if all_votes:
      all_votes[0]._debug_values = {
        'delegations': delegations,
        'known_votes': known_votes,
        'normalized_votes': normalized_votes,
      }

    return all_votes

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

def main():
  """
  Main function.

  It makes a random population, a random delegation network with random votes and computes results for all this.
  """

  for seed in range(0, 1000):
    # We initialize the random generator to a constant so that runs are reproducible.
    random.seed(seed)

    # Size of a random population.
    size = 4

    # Random population.
    persons = [Person() for i in range(size)]

    # We define random delegates.
    for p in persons:
      sample = random.sample(persons, random.randint(0, int(math.sqrt(size))))
      delegates = [Delegate(s, random.uniform(0, 1)) for s in sample if s is not p]
      sum = 1e-12
      for s in delegates:
        sum += s.ratio
      for s in delegates:
        s.ratio /= sum
      p.delegates(delegates)

    # And some from population randomly vote
    random_sample = random.sample(persons, random.randint(1, size / 2))
    votes = sorted([Vote(p, random.uniform(-1, 1)) for p in random_sample], key=lambda el: el.person)

    results = []
    results_votes = []
    for cls in (RecursiveDelegationOne, RecursiveDelegationTwo, LinearDelegation):
      before = time.clock()
      all_votes = cls.compute_all_votes(persons, votes)
      after = time.clock()
      results.append(compute_results(all_votes))
      results_votes.append(all_votes)
      #print u"Result: %.2f, time: %.3fs" % (results[-1], after - before)

    if not checkEqual(results):
      print results

      print u"Delegations:"

      for p in persons:
        print u" %s:" % p.name
        for s in p.delegates():
          print u"  %s" % s

      print u"Votes:"

      for v in votes:
        print u" %s" % v

      for i, all_votes in enumerate(results_votes):
        print u"Delegated votes %s:" % i

        for v in sorted(all_votes):
          print u" %s" % v
          if v._debug_values is not None:
            print u"  %s" % v._debug_values

if __name__ == "__main__":
  main()
