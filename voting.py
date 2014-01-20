import math
import random

class Base:
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
		# Default is to trust yourself ultimately (but it could also be 1 here)
		self._delegates = [Delegate(self, float('inf'))]

	def __unicode__(self):
		"""
		Returns this person's unicode representation, her name.
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
			sum = 0
			missing_self = True
			delegates_dict = {}
			for s in delegates:
				if s.ratio == float('inf'):
					if s.person != self:
						# Only a person herself can have a precedence before delegates
						raise ValueError("Invalid ratio: %f" % s.ratio)
				elif s.ratio <= 0 or s.ratio > 1:
					raise ValueError("Invalid ratio: %f" % s.ratio)
				else:
					sum += s.ratio
				if s.person == self:
					missing_self = False
				if s.person in delegates_dict:
					delegates_dict[s.person].ratio += s.ratio
				else:
					delegates_dict[s.person] = s
			if missing_self:
				raise ValueError("Delegates do not contain person herself.")
			if abs(1 - sum) > 1e-10 and not (len(delegates_dict) == 1 and \
					delegates_dict.values()[0].ratio == float('inf')):
				raise ValueError("Sum of all ratios is not 1 but %f" % sum)
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
		@param ratio: A ratio of trust for this delegate. It has to be a number from (0, 1] or
		              +infinity (only in the case a person and her surogate are the same person).
		"""

		if (ratio <= 0 or ratio > 1) and ratio != float('inf'):
			raise ValueError("Invalid ratio: %f" % ratio)

		self.person = person
		self.ratio = ratio

	def __unicode__(self):
		"""
		Returns this delegate's unicode representation, her and her ratio.
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

	def __init__(self, person, option, ratio=1):
		"""
		Class constructor.

		@param person: Person this vote is from.
		@param option: Which option this vote is for.
		@param ratio: Ratio of option the vote is for. It has to be a number from (0, 1].
		"""

		if ratio <= 0 or ratio > 1:
			raise ValueError("Invalid ratio: %f" % ratio)

		self.person = person
		self.option = option
		self.ratio = ratio

	def __unicode__(self):
		"""
		Returns this vote's unicode representation, option and its ratio.
		"""

		return u"%s(%.2f)" % (self.option, self.ratio)

def delegate_vote(person, votes_dict, pending, visited=[]):
	"""
	For persons who did not vote for themselves this function computes a delegate vote.

	@param person: A person to computer delegate vote for.
	@param votes_dict: A dictionary of already computer votes.
	@param pending: A dictionary of pending persons to computer delegate vote for.
	@param visited: A list of delegate votes we have already tried to compute.

	@return: None if the delegate vote for this person is impossible to compute, an empty list
	         if it is currently not possible to compute it or a list of computed votes
	"""

	if person in votes_dict:
		# Vote for this person is already known
		return votes_dict[person]

	if person not in pending:
		# This person does not exist in current population so we are unable to compute vote her
		# This probably means that we removed the person from votes_dict and pending as her delegate
		# vote was not computable
		return None

	delegates = person.delegates()
	assert len(delegates) > 0

	if len(delegates) == 1:
		# Person does not have any delegates defined (except herself)
		assert delegates[0].person == person
		return None
	if person in visited:
		# In computing the delegate vote we came back to vote we are already computing 
		return None

	votes = []
	for s in delegates:
		if s.person == person:
			assert True if pending[s.person] is None else s.ratio != float('inf')
			votes.append((s.ratio, pending[s.person]))
		else:
			votes.append((s.ratio, delegate_vote(s.person, votes_dict, pending, visited + [person])))
	
	known_votes = [(r, vs) for (r, vs) in votes if vs is not None]

	if len(known_votes) == 0:
		# The delegate vote is impossible to compute for this person
		# (we have a subgraph where we cannot do anything)
		return None

	sum = 0
	for (r, vs) in known_votes:
		if vs == []:
			# The delegate is currently not possible to compute
			return []
		else:
			sum += r
	
	known_votes = [(r/sum, vs) for (r, vs) in known_votes] # We normalize
	
	results = {}
	for (r, vs) in known_votes:
		for v in vs:
			results[v.option] = results.get(v.option, 0) + r * v.ratio

	return [Vote(person, option, ratio) for (option, ratio) in results.items()]

def delegate_version(persons, options, votes):
	"""
	Delegate version of calculating results.

	@param persons: population
	@param options: options from which votes were made
	@param votes: votes made

	@return: A dictionary of options and votes for this options
	"""

	if len(votes) == 0:
		raise ValueError("At least one vote has to be cast")
	if len(options) < 2:
		raise ValueError("We have to have options to choose from")
	if len(persons) == 0:
		raise ValueError("Zero-sized population")

	# A dictionary of (currently) finalized votes, each person can have multiple votes,
	# but the sum of all her votes has to be 1
	votes_dict = {}

	# Persons we have to calculate something for
	# (who didn't vote or who do not use precedence before delegates)
	pending = {}

	for v in votes:
		assert v.ratio == 1
		assert v.person not in votes_dict and v.person not in pending

		# Person has voted and has a precedence before delegates or it is the only one delegate defined
		if v.person.delegates()[0].ratio == float('inf') or len(v.person.delegates()) == 1:
			assert v.person.delegates()[0].person == v.person
			votes_dict[v.person] = [v]
		else:
			pending[v.person] = [v]
	for p in persons:
		if not (p in votes_dict.keys() or p in pending.keys()):
			pending[p] = None

	# A list of persons we could not compute votes for
	unable_votes = []

	while len(pending) > 0:
		for p in pending.keys():
			assert p not in votes_dict and p not in unable_votes
			vs = delegate_vote(p, votes_dict, pending)
			if vs is None:
				# This one we will never be able to compute
				unable_votes.append(p)
				del pending[p]
			elif vs:
				assert all([v.person == p for v in vs])
				assert abs(1 - sum([v.ratio for v in vs])) < 1e-10
				votes_dict[p] = vs
				del pending[p]
			# Otherwise we cannot do anything at this moment with this pending person

	# Sums the results
	results_dict = dict([(o, 0) for o in options])
	for vs in votes_dict.values():
		assert vs
		for v in vs:
			results_dict[v.option] += v.ratio
	
	if unable_votes:
		print "Unable to compute vote for: %s" % ", ".join(["%s" % s for s in sorted(unable_votes)])
	
	return sorted(results_dict.items())

def main():
	"""
	Main function.

	It makes random population, random trust network with random votes and computes results for all this.
	"""

	size = 4 # Size of a random population
	options = ['A', 'B'] # Options for random votes

	persons = [Person() for i in range(size)] # Random population

	# We define random delegates
	for p in persons:
		sample = random.sample(persons, random.randint(0, int(math.sqrt(size))))
		delegates = [Delegate(s, 1 - random.random()) for s in sample]
		sum = 0
		for s in delegates:
			sum += s.ratio
		for s in delegates:
			s.ratio /= sum
		if p not in sample:
			delegates.append(Delegate(p, float('inf'))) # Person herself has to be among delegates
		p.delegates(delegates)
	
	# And some from population randomly vote
	random_sample = random.sample(persons, random.randint(1, size / 2))
	votes = sorted([Vote(p, random.choice(options)) for p in random_sample], key=lambda el: el.person)
	
	for p in persons:
		print u"%s:" % p.name
		for s in p.delegates():
			print u" %s" % s

	for v in votes:
		print u"%s: %s" % (v.person, v)

	# What does delegate version of results say?
	print delegate_version(persons, options, votes)

if __name__ == "__main__":
	main()
