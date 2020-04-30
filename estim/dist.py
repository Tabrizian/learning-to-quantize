from scipy.stats import truncnorm, norm
import numpy as np

class Normal:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
    
    def cdf(self, x):
        return norm.cdf(x, loc=self.mean, scale=self.sigma)

    def pdf(self, x):
        return norm.pdf(x, loc=self.mean, scale=self.sigma)

class TruncNorm:
    def __init__(self, mean, sigma, begin, end):
        self.mean = mean
        self.sigma = sigma
        self.a, self.b = (begin - mean) / sigma, (end - mean) / sigma

    def cdf(self, x):
        return truncnorm.cdf(x, self.a, self.b, loc=self.mean, scale=self.sigma)

    def pdf(self, x):
        return truncnorm.pdf(x, self.a, self.b, loc=self.mean, scale=self.sigma)

    def ppf(self, x):
        return truncnorm.ppf(x, self.a, self.b, loc=self.mean, scale=self.sigma)

class CondNormal:
    def __init__(self, means, sigmas, norms):
        self.means = means
        self.sigmas = sigmas
        self.norms = norms
        self.total_norm = 0.0
        for norm in norms:
            self.total_norm += norm

    def cdf(self, x):
        result = 0.0
        for i in range(len(means)):
            result += norms[i] / self.total_norm * norm.cdf(x, loc=self.means[i], scale=self.sigmas[i])

        return result

    def pdf(self, x):
        result = 0.0
        for i in range(len(means)):
            result += norms[i] / self.total_norm * norm.pdf(x, loc=self.means[i], scale=self.sigmas[i])

        return result


class CondNormalTrunc:
    def __init__(self, means, sigmas, norms, begin, end):
        self.means = np.asarray(means)
        self.sigmas = np.asarray(sigmas)
        self.norms = np.asarray(norms)
        self.end = end
        self.total_norm = np.sum(self.norms)
        
        self.a = (begin - self.means) / self.sigmas
        self.b = (end - self.means) / self.sigmas
        self.coeff = self.norms / self.total_norm

    def cdf(self, x):
        cdfs = truncnorm.cdf(x, self.a, self.b, loc=self.means, scale=self.sigmas)
        return np.sum(np.dot(cdfs, self.coeff))

    def pdf(self, x):
        pdfs = truncnorm.pdf(x, self.a, self.b, loc=self.means, scale=self.sigmas)
        return np.sum(np.dot(pdfs, self.coeff))