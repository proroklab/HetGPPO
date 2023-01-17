#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import numpy as np
import scipy
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

    sqrtK_1 = scipy.linalg.sqrtm(K_1)
    first_term = sqrtK_1 @ K_0
    K_1_K_0_K_1 = first_term @ sqrtK_1

    cov_dist = (
        np.trace(K_0)
        + np.trace(K_1)
        - 2 * np.trace(scipy.linalg.sqrtm(K_1_K_0_K_1).astype(np.float32))
    )

    l2norm = np.linalg.norm(mu_0 - mu_1, ord=2) ** 2

    d = np.real(np.sqrt(l2norm + cov_dist))
    if d < 0:
        print(d)
        raise AssertionError

    return d


def kl_divergence(mean, sigma, mean2, sigma2):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.


    From wikipedia
    KL( (mean, sigma) || (mean2, sigma2))
         = .5 * ( tr(sigma2^{-1} sigma) + log |sigma2|/|sigma| +
                  (mean2 - mean)^T sigma2^{-1} (mean2 - mean) - N )
    """
    # store inv diag covariance of sigma2 and diff between means
    N = mean.shape[0]
    isigma2 = np.linalg.inv(sigma2)
    diff = mean2 - mean

    # kl is made of three terms
    tr_term = np.trace(isigma2 @ sigma)
    det_term = np.log(
        np.linalg.det(sigma2) / np.linalg.det(sigma)
    )  # np.sum(np.log(sigma2)) - np.sum(np.log(sigma))
    quad_term = diff.T @ np.linalg.inv(sigma2) @ diff
    kl = 0.5 * (tr_term + det_term + quad_term - N)
    assert kl > -1e-3
    return max(0.0, kl)


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

    b_d = bhattacharyya_distance(mu1, sigma1, mu2, sigma2)
    bc = np.exp(-b_d)

    assert hellinger == np.sqrt(1 - bc)

    return hellinger


def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):

    m1_minus_m2 = mu1 - mu2
    sigma = (sigma1 + sigma2) / 2

    det_sigma1 = np.linalg.det(sigma1)
    det_sigma2 = np.linalg.det(sigma2)
    det_sigma = np.linalg.det(sigma)

    det_term = 0.5 * np.log(det_sigma / np.sqrt(det_sigma1 * det_sigma2))

    central_term = m1_minus_m2.T @ np.linalg.inv(sigma) @ m1_minus_m2

    d = (1 / 8) * central_term + det_term

    return d


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
