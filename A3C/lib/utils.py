
import math


def log_uniform(lo, hi, rate):
	log_lo = math.log(lo)
	log_hi = math.log(hi)
	v = log_lo * (1-rate) + log_hi * rate
	return math.exp(v)
