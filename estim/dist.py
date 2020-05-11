from scipy.stats import truncnorm
from scipy import integrate
import numpy as np
import bisect
import matplotlib.pyplot as plt


class Distribution:

    def __init__(self, begin=-1, end=+1, nbins=1000, bin_type='linear'):
        self.begin = begin
        self.end = end
        self.bin_edges = bin_edges = self._get_bin_edges(nbins, bin_type)
        self.bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        self.bin_width = (bin_edges[1:]-bin_edges[:-1])

    def _get_bin_edges(self, nbins, bin_type):
        if bin_type == 'linear':
            bin_edges = np.linspace(self.begin, self.end, nbins)
        elif bin_type == 'log':
            bin_edges = np.logspace(self.begin, self.end, nbins)/10
            bin_edges = np.concatenate((-np.flip(bin_edges), [0], bin_edges))
            # TODO: assumes symmetric around 0
        return bin_edges

    def est_var_adjacent_levels(self, left_level, right_level):
        # From Eq 6 in the paper
        # int_a^b sigma^2(r) f(r) dr
        # = sum_{ind(e_l)}^{ind(e_r)} f(r)
        #         int_{max(a,e_l)}^{min(e_r,b)} sigma^2(r) dr
        # TODO: test this function
        bin_edges = self.bin_edges
        # TODO: test these bisects
        # left_edge = bisect.bisect_right(bin_edges, left_level)-1
        # right_edge = bisect.bisect_left(bin_edges, right_level)
        var = 0

        c = left_level
        d = right_level

        # for index in range(left_edge, right_edge):
        #     a = max(bin_edges[index], left_level)
        #     b = min(bin_edges[index+1], right_level)
        #     # int_a^b (x - c) (d - x) dx = 1/6 (a - b) (2 a^2 + 2 a b
        #     #         - 3 a (c + d) + 2 b^2 - 3 b (c + d) + 6 c d)
        #     #         where c is left_level and d is right_level
        #     center = (a+b)/2
        #     var += self.pdf(center) * 1/6 * (a-b) * (
        #         2*a**2+2*a*b-3*a*(c+d)+2*b**2-3*b*(c+d)+6*c*d)

        def f(x):
            return (x - c) * (d - x) * self.pdf(x)

        # print('adj_lev', left_level, right_level)
        intg = integrate.quad(f, c, d)[0]
        # x = np.linspace(c, d, 10000)
        # y = [f(val_x) for val_x in x]
        # plt.plot(x, y)
        # plt.show()
        # assert np.abs(intg - var) < 1e-5

        return intg

    def estimate_variance_adj_inv(self, left_level, right_level):
        # calculate Eq 8 of the paper
        # ppf(cdf(d) - int_c^d (r - c) * pdf(r) dr / (d - c))
        # integration is equal to
        # = sum_{ind(e_l)}^{ind(e_r)} f(r)
        #         int_{max(a,e_l)}^{min(e_r,b)} (r-c) dr
        # where c is left_level and d is right_level
        bin_edges = self.bin_edges
        # left_edge = bisect.bisect_right(bin_edges, left_level)-1
        # right_edge = bisect.bisect_left(bin_edges, right_level)
        intg = 0
        c = left_level
        d = right_level

        # for index in range(left_edge, right_edge):
        #     a = max(bin_edges[index], left_level)
        #     b = min(bin_edges[index+1], right_level)
        #     # int_a^b (r - c) dr = -1/2 (a - b) (a + b - 2 c)
        #     # where c is left_level and d is right_level
        #     center = (a+b)/2
        #     intg += self.pdf(center) * -1/2 * (a - b) * (a + b - 2 * c)

        def f(x):
            return (x - c) * self.pdf(x)
        # print('adj_lev_inv', left_level, right_level)
        # x = np.linspace(c, d, 10000)
        # y = [f(val_x) for val_x in x]
        # plt.plot(x, y)
        # plt.show()
        intg_by_intg = integrate.quad(f, c, d)[0]
        # err = np.abs(intg - intg_by_intg)

        #assert err < 1e-4, \
        #    'Integration does not have enough accuracy left level %s, right level %s, error is %s' \
        #    % (left_level, right_level, err)

        inv_arg = self.cdf(right_level) - intg_by_intg / (d-c)
        return self.ppf(inv_arg)

    def estimate_variance(self, levels):
        # TODO: test this function
        var = 0
        for index, left_level in enumerate(levels[:-1]):
            right_level = levels[index+1]
            var += self.est_var_adjacent_levels(
                left_level, right_level)
        return var

    def estimate_variance_int(self, levels, dist=None):
        # variance estimate calculation by integration
        # optional dist parameter to provide your own distribution function

        var = 0.0
        dist = self if dist is None else dist

        for index, _ in enumerate(levels[:-1]):
            def f(x):
                pdf = dist.pdf(x)
                index_l = bisect.bisect_left(levels, x) - 1
                variance = (x - levels[index_l]) * (levels[index_l + 1] - x)
                return variance * pdf
            var += integrate.quad(lambda x: f(x),
                                  levels[index], levels[index + 1])[0]
        return var

    def pdf(self, x):
        raise NotImplementedError('PDF has not been implemented.')

    def cdf(self, x):
        raise NotImplementedError('CDF has not been implemented.')


class TruncNorm(Distribution):

    def __init__(self, mean, sigma, begin=-1, end=+1, nbins=100,
                 bin_type='linear'):
        super().__init__(begin, end, nbins, bin_type)

        self.mean = mean
        self.sigma = sigma
        self.begin = begin
        self.end = end
        self.nbins = nbins
        self.a = (begin - self.mean) / self.sigma
        self.b = (end - self.mean) / self.sigma

    def cdf(self, x):
        a = self.a
        b = self.b
        mu = self.mean
        sigma = self.sigma
        return truncnorm.cdf(x, a, b, loc=mu, scale=sigma)

    def pdf(self, x):
        a = self.a
        b = self.b
        mu = self.mean
        sigma = self.sigma
        return truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

    def ppf(self, x):
        a = self.a
        b = self.b
        mu = self.mean
        sigma = self.sigma
        return truncnorm.ppf(x, a, b, loc=mu, scale=sigma)


class CondNormalTrunc(Distribution):

    def __init__(self, means, sigmas, norms, begin=-1, end=1,
                 nbins=1000, bin_type='linear'):
        super().__init__(begin, end, nbins, bin_type)
        self.means = np.asarray(means)
        self.sigmas = np.asarray(sigmas)
        self.norms = np.asarray(norms)
        self.end = end
        self.begin = begin
        self.total_norm = np.sum(self.norms)

        self.a = (begin - self.means) / self.sigmas
        self.b = (end - self.means) / self.sigmas
        self.coeff = self.norms / self.total_norm

    def cdf(self, x):
        cdfs = truncnorm.cdf(
            x, self.a, self.b, loc=self.means, scale=self.sigmas)
        return np.sum(np.dot(cdfs, self.coeff))

    def pdf(self, x):
        pdfs = truncnorm.pdf(
            x, self.a, self.b, loc=self.means, scale=self.sigmas)
        return np.sum(np.dot(pdfs, self.coeff))


class CondNormalTruncHist(Distribution):

    def __init__(self, means, sigmas, norms, begin=-1, end=+1, nbins=100,
                 bin_type='linear'):
        super().__init__(begin, end, nbins, bin_type)
        self.means = np.asarray(means)
        self.sigmas = np.asarray(sigmas)
        self.norms = np.asarray(norms)
        self.begin = begin
        self.end = end
        self.nbins = nbins
        self.total_norm = np.sum(self.norms)
        self.a = (begin - self.means) / self.sigmas
        self.b = (end - self.means) / self.sigmas
        self.coeff = self.norms / self.total_norm

        self.pdf_bin_sum = self._quantized_sum_pdf()
        self.cdf_bin_sum = np.cumsum(self.pdf_bin_sum).clip(0, 1)
        # self.ppf_bin_width = (self.cdf_bin_sum[1:]-self.cdf_bin_sum[:-1])
        self.pdf_at_centers = self.pdf_bin_sum / self.bin_width

    def _quantized_sum_pdf(self):
        from scipy import stats
        mu = self.means
        sigma = self.sigmas
        norms = self.norms
        a_vals = self.a
        b_vals = self.b
        bin_edges = self.bin_edges
        pdf_bin_sum = 0
        for m, s, n, a_val, b_val in zip(mu, sigma, norms, a_vals, b_vals):
            cdfa = stats.truncnorm.cdf(bin_edges[:-1], loc=m, scale=s,
                                       a=a_val, b=b_val)
            cdfb = stats.truncnorm.cdf(bin_edges[1:], loc=m, scale=s,
                                       a=a_val, b=b_val)
            pdfb = cdfb-cdfa
            pdfb /= pdfb.sum()
            pdf_bin_sum = n / self.total_norm * pdfb + pdf_bin_sum
        pdf_bin_sum /= pdf_bin_sum.sum()
        return pdf_bin_sum

    def cdf(self, x):
        index = bisect.bisect_right(self.bin_edges, x)-1
        if index == len(self.bin_edges)-1:
            # case: x=self.end
            return 1.0
        cdf_at_x = self.cdf_bin_sum[index-1] if index > 0 else 0
        weight = (x-self.bin_edges[index])/self.bin_width[index]
        cdf_at_x += weight*self.pdf_bin_sum[index]
        return cdf_at_x

    def pdf(self, x):
        index = bisect.bisect_right(self.bin_edges, x)-1
        if index == len(self.pdf_at_centers):
            return 0.0
        return self.pdf_at_centers[index]

    def ppf(self, cdf_at_x):
        # TODO: I need to test
        index = bisect.bisect_right(self.cdf_bin_sum, cdf_at_x)-1
        if index == len(self.cdf_bin_sum)-1:
            # case: cdf_at_x = 1
            return 1.0
        # TODO: should we set to self.begin or self.a?
        # special case: left edge
        x = self.bin_edges[index] if index >= 0 else self.begin
        ppf_bin_width = self.cdf_bin_sum[index+1]-self.cdf_bin_sum[index]
        weight = (cdf_at_x-self.cdf_bin_sum[index])/ppf_bin_width
        x += weight*self.bin_width[index]
        return x
