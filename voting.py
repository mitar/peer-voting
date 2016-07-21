#!/usr/bin/env python

import math
import random

# We initialize the random generator to a constant so that runs are reproducible.
random.seed(42)

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

  def __unicode__(self):
    """
    Returns this vote's unicode representation.
    """

    return u"%.2f" % self.vote

def delegate_vote(person, votes_dict, pending, visited=[]):
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

    votes.append((d.ratio, delegate_vote(d.person, votes_dict, pending, visited + [person])))
  
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

  # TODO: Store computed vote into votes_dict and remove it from pending.
  return Vote(person, result)

def compute_all_votes(persons, votes):
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

      vote = delegate_vote(p, votes_dict, pending)

      if vote is None:
        # This one we will never be able to compute.
        unable_votes.append(p)
      else:
        # TODO: This could be done by delegate_vote itself.
        votes_dict[p] = vote

      pending.remove(p)

      # We just want to pick an arbitrary p from pending in this inner loop.
      # But we cannot continue looping in inner loop because we modified the pending set.
      break

  if unable_votes:
    print "Unable to compute vote for: %s" % ", ".join(["%s" % s for s in sorted(unable_votes)])

  return votes_dict.values()

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

def main():
  """
  Main function.

  It makes a random population, a random delegation network with random votes and computes results for all this.
  """

  # Size of a random population.
  size = 50

  # Random population.
  persons = [Person() for i in range(size)]

  # We define random delegates.
  for p in persons:
    sample = random.sample(persons, random.randint(0, int(math.sqrt(size))))
    delegates = [Delegate(s, random.uniform(0, 1)) for s in sample if s is not p]
    sum = 1e-15
    for s in delegates:
      sum += s.ratio
    for s in delegates:
      s.ratio /= sum
    p.delegates(delegates)

  # And some from population randomly vote
  random_sample = random.sample(persons, random.randint(1, size / 2))
  votes = sorted([Vote(p, random.uniform(-1, 1)) for p in random_sample], key=lambda el: el.person)

  for p in persons:
    print u"%s:" % p.name
    for s in p.delegates():
      print u" %s" % s

  for v in votes:
    print u"%s: %s" % (v.person, v)

  votes = compute_all_votes(persons, votes)
  print u"Result: %.2f" % compute_results(votes)

if __name__ == "__main__":
  main()
