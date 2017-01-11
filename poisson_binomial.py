 #!/usr/bin/python
# -*- coding: utf-8 -*-

""" a class for different poisson binomial calculations, in all functions ps is expected to be a list with the probabilities"""
import operator
import functools
import math
import numpy as np
import scipy

def mu(ps):
	return sum(ps)

def std(ps):
	ps = np.array(ps)
	return np.sqrt(np.dot(ps,1-ps))

def pval(ps, x, approximate=False):
	"""returns the probability given ps that x or a bigger value will be seen"""
	return mass_for_poisson_binomial_probability_range(ps, range(math.ceil(x), len(ps)), approximate)

def poisson_binomial_PMF_possion_approximation(ps, n):
	mu = mu(ps)
	return (mu**n)*math.exp(-mu)/math.factorial(n)

def poisson_binomial_CDF_refined_normal_approximation(ps, n):
	""" RNA approximation based on 
	Neammanee, K. (2005).
	A refinement of normal approximation to Poisson binomial.
	International Journal of Mathematics and Mathematical Sciences, 5, 717–728.
	"""
	ps = np.array(ps)
	mu = mu(ps)
	std = std(ps)
	gamma = np.power(std, -3)*np.dot(np.multiply(ps,1-ps), 1-2*ps)
	x = (n + 0.5 - mu) / std
	phi_x = scipy.stats.norm(0, 1).pdf(x)
	big_phi_x = scipy.stats.norm(0, 1).cdf(x)
	return big_phi_x + (gamma*(1-x**2)*phi_x)/6

def poisson_binomial_CDF_normal_approximation(ps, n):
	"""Based on central theorem"""
	ps = np.array(ps)
	mu = mu(ps)
	std = std(ps)
	gamma = np.power(std, -3)*np.dot(np.multiply(ps,1-ps), 1-2*ps)
	x = (n + 0.5 - mu) / std
	return scipy.stats.norm(0, 1).cdf(x)

def poisson_binomial_PMF_DFT(ps, n, emptyCache=False):
	return poisson_binomial_PMFS_DFT(ps, emptyCache)[n]

def poisson_binomial_CDF_DFT(ps, n, emptyCache=False):
	return np.sum(poisson_binomial_PMFS_DFT(ps, emptyCache)[:n])

dft_cache = {}
def poisson_binomial_PMFS_DFT(ps, emptyCache=False):
	""" calculates the whole distribution using DFT method
		proposed by 
		Hong, Yili. 
		"On computing the distribution function for the Poisson binomial distribution."
		Computational Statistics & Data Analysis 59 (2013): 41-51.
		ps - an iterable of p values representing the distribution
		emptyCache - If True empties the cache
					 Should be checked if the user wishes no cache to be used,
					  or wishes to discard the caching done so far 
					  (e.g. when ps are not going to rpeat themselves).
	"""
	global dft_cache
	if emptyCache:
		dft_cache = {}
	ps = np.array(ps)
	hashable = tuple(ps)
	if hashable not in dft_cache:
		ps_num = len(ps)
		omega = (2*math.pi)/(ps_num + 1)
		a = np.ones((ps_num+1, ))
		b = np.zeros((ps_num+1, ))
		for l in range(1, math.ceil(ps_num/2)+1):
			zl = 1-ps + ps*np.cos(omega*l) + np.complex(0,1)*ps*np.sin(omega*l)		
			dl = math.exp(np.sum(np.log(np.absolute(zl))))
			arg_zl = np.angle(zl)
			sum_args = np.sum(arg_zl)
			a[l] = dl*math.cos(sum_args)
			b[l] = dl*math.sin(sum_args)
		for l in range(math.ceil(ps_num/2)+1, ps_num + 1):
			a[l] = a[ps_num +1 -l]
			b[l] = -b[ps_num +1 -l]
		x = a + 1j * b
		result = np.fft.fft(x / (ps_num + 1))

		dft_cache[hashable] = np.real(result)
	return dft_cache[hashable]

def poisson_binomial_PMF(ps, n):
	"""calculates the exact probability of n from a poisson binomial distribution, charachterized by ps
		uses the faster recursive way as evaluated by
		Chen, Sean X., and Jun S. Liu.
		"Statistical applications of the Poisson-binomial and conditional Bernoulli distributions."
		Statistica Sinica (1997): 875-892.
		the way was suggested in
		Chen, X. H., Dempster, A. P. and Liu, J. S. (1994).
		Weighted finite population sampling to
		maximize entropy. Biometrika 81, 457-469."""
	# note (1+w)**-1 = 1-p
	return functools.reduce(operator.mul, (1-x for x in ps), poisson_binomial_R(ps, n, range(len(ps))))


wis = {}
def poisson_binomial_w_times_i(p, i=1):
	if p not in wis:
		wis[p] = [1, p/(1-p)]

	lst = wis[p]
	for place in range(len(lst) - i + 1):
		lst.append(lst[1]*lst[-1])
	return lst[i] 


def poisson_binomial_T(ps, i, chosen):
	return np.sum((poisson_binomial_w_times_i(ps[choice],i) for choice in chosen))


def poisson_binomial_R(ps, n, chosen):
	if len(chosen) < n:
		return 0
	if not n:
		return 1
	# pool = Pool(10) #TODO is pooling needed? and working despite recursion
	a = np.empty((n,),int)
	a[::2] = 1
	a[1::2] = -1
	it = []
	# it += list(pool.imap(__compute_coverage, range(1,n+1)))
	it = np.fromiter((poisson_binomial_T(ps,i,chosen)*poisson_binomial_R(ps, n-i, chosen) for i in range(1,n+1)), np.float)
	# pool.close()
	# pool.join()
	return np.inner(it, a)/n
