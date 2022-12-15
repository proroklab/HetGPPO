#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import numpy as np
import scipy
import scipy.linalg
import torch.distributions


def wasserstein_distance(mean, sigma, mean2, sigma2):
    # ensure that means are 1D arrays
    mean = mean.reshape(-1)
    mean2 = mean2.reshape(-1)

    # check that covariances are positive definite
    # by computing matrix square roots
    SC = np.linalg.cholesky(sigma)
    SC2 = np.linalg.cholesky(sigma2)

    return np.sqrt(
        np.linalg.norm(mean - mean2, ord=2) ** 2
        + np.linalg.norm(SC - SC2, ord="fro") ** 2
    )


def wasserstein_distance2(mean, sigma, mean2, sigma2):
    mu_0 = mean
    K_0 = sigma
    mu_1 = mean2
    K_1 = sigma2

    sqrtK_0 = scipy.linalg.sqrtm(K_0)
    first_term = sqrtK_0 @ K_1
    K_0_K_1_K_0 = first_term @ sqrtK_0

    cov_dist = (
        np.trace(K_0)
        + np.trace(K_1)
        - 2 * np.trace(scipy.linalg.sqrtm(K_0_K_1_K_0).astype(np.float32))
    )

    l2norm = np.linalg.norm(mu_0 - mu_1, ord=2) ** 2

    d = np.real(np.sqrt(l2norm + cov_dist))
    assert d >= 0

    return d


def kl_divergence(mean, sigma, mean2, sigma2, eps=1e-12):
    # ensure that means are 1D arrays
    mean = mean.reshape(-1)
    mean2 = mean2.reshape(-1)

    # check that covariances are positive definite
    np.linalg.cholesky(sigma)
    np.linalg.cholesky(sigma2)

    # compute difference in means
    meandiff = mean2 - mean

    # ignoring tiny differences in dimensions with tiny variance
    var = sigma.diagonal()
    var2 = sigma2.diagonal()
    tinyind = ((var < eps) * (var2 < eps)).nonzero()[0]
    if tinyind.size > 0:
        if meandiff[tinyind] < eps:
            meandiff[tinyind] = 0.0

    # compute inverse of Sigma2
    Sigma2inv = np.linalg.inv(sigma2)

    # compute log determinants
    (sign, logdet) = np.linalg.slogdet(sigma)
    sldet = sign * logdet
    (sign, logdet) = np.linalg.slogdet(sigma2)
    sldet2 = sign * logdet

    return 0.5 * (
        (Sigma2inv @ sigma).trace()
        - mean.size
        + meandiff @ Sigma2inv @ meandiff
        + sldet2
        - sldet
    )


def kl_symmetric(mean, sigma, mean2, sigma2):
    return kl_divergence(mean, sigma, mean2, sigma2) + kl_divergence(
        mean2, sigma2, mean, sigma
    )


def hellinger_distance(mu1, sigma1, mu2, sigma2):

    sigma1_plus_sigma2 = sigma1 + sigma2
    mu1_minus_mu2 = mu1 - mu2

    E = mu1_minus_mu2.T @ np.linalg.inv(sigma1_plus_sigma2 / 2) @ mu1_minus_mu2
    epsilon = -(1 / 8) * E

    numerator = np.sqrt(np.linalg.det(sigma1 @ sigma2))
    denominator = np.linalg.det(sigma1_plus_sigma2 / 2)

    squared_hellinger = 1 - np.sqrt(numerator / denominator) * np.exp(epsilon)
    hellinger = np.sqrt(squared_hellinger).item()

    assert 0 <= hellinger <= 1

    return hellinger


def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    hellinger = hellinger_distance(mu1, sigma1, mu2, sigma2)
    bhat_coeff = 1 - (hellinger**2)

    assert 0 <= bhat_coeff <= 1

    bhat_dist = -np.log(bhat_coeff)

    assert bhat_dist >= 0

    return np.abs(bhat_dist)


if __name__ == "__main__":

    for j in range(1, 25):
        for i in range(1, 25):

            dist_1 = torch.distributions.MultivariateNormal(
                torch.tensor([float(i)]), torch.tensor([[float(j)]])
            )
            dist_2 = torch.distributions.MultivariateNormal(
                torch.tensor([float(j)]), torch.tensor([[float(i)]])
            )
            print(dist_1.loc.shape)

            print(
                bhattacharyya_distance(
                    dist_1.loc.numpy(),
                    dist_1.covariance_matrix.numpy(),
                    dist_2.loc.numpy(),
                    dist_2.covariance_matrix.numpy(),
                )
            )

            assert round(
                bhattacharyya_distance(
                    dist_1.loc.numpy(),
                    dist_1.covariance_matrix.numpy(),
                    dist_2.loc.numpy(),
                    dist_2.covariance_matrix.numpy(),
                ),
                4,
            ) == round(
                bhattacharyya_distance(
                    dist_2.loc.numpy(),
                    dist_2.covariance_matrix.numpy(),
                    dist_1.loc.numpy(),
                    dist_1.covariance_matrix.numpy(),
                ),
                4,
            )
