# Dataset-level Fairness Metrics

We provide here a list of the various metrics available for measuring fairness of
datasets, as well as a short description of each of them.

1. **Metrics that compare against a given reference distribution:** These metrics involve computing
some measure of distance or divergence from a given reference distribution provided by the user.
The library supports only the `UNIFORM` distribution out of the box (all `label-protectedAttribute`
combinations must have equal number of records), but users may supply their own distribution
(such as an apriori known gender distribution etc.).

    For the most up-to-date documentation on the supported metrics, please look at the link [here](lift/src/main/scala/com/linkedin/lift/lib/DivergenceUtils.scala), and look for the
    `computeDatasetDistanceMetrics` method as the starting point. The following metrics fall under this
    category:

    1. **Skews:** Computes the logarithm of the ratio of the observed value to the expected value. For example, if we are dealing with label-gender distributions, this metric computes

        ![\log\left(\frac{(0.0, MALE)_{obs}}{(0.0, MALE)_{exp}}\right), \log\left(\frac{(1.0, MALE)_{obs}}{(1.0, MALE)_{exp}}\right), \log\left(\frac{(0.0, FEMALE)_{obs}}{(0.0, FEMALE)_{exp}}\right), \log\left(\frac{(1.0, FEMALE)_{obs}}{(1.0, FEMALE)_{exp}}\right)](https://render.githubusercontent.com/render/math?math=%5Clog%5Cleft(%5Cfrac%7B(0.0%2C%20MALE)_%7Bobs%7D%7D%7B(0.0%2C%20MALE)_%7Bexp%7D%7D%5Cright)%2C%20%5Clog%5Cleft(%5Cfrac%7B(1.0%2C%20MALE)_%7Bobs%7D%7D%7B(1.0%2C%20MALE)_%7Bexp%7D%7D%5Cright)%2C%20%5Clog%5Cleft(%5Cfrac%7B(0.0%2C%20FEMALE)_%7Bobs%7D%7D%7B(0.0%2C%20FEMALE)_%7Bexp%7D%7D%5Cright)%2C%20%5Clog%5Cleft(%5Cfrac%7B(1.0%2C%20FEMALE)_%7Bobs%7D%7D%7B(1.0%2C%20FEMALE)_%7Bexp%7D%7D%5Cright))

    2. **Infinity Norm Distance:** Computes the Chebyshev Distance between the observed and reference distribution. It equals the maximum difference between the two distributions.
    3. **Total Variation Distance:** Computes the Total Variation Distance between the observed and reference distribution. It is equal to half the L1 distance between the two distributions.
    4. **JS Divergence:** The Jensen-Shannon Divergence between the observed and reference distribution. Suppose that the average of these two distributions is given by M. Then, the JS Divergence is the average of the KL Divergences between the observed distribution and M, and the reference distribution and M.
    5. **KL Divergence:** The Kullback-Leibler Divergence between the observed and reference distribution. It is the expectation (over the observed distribution) of the logarithmic differences between the observed and reference distributions. The latter is the Skew we measure above.

2. **Metrics computed on the observed distribution only:** These metrics compute some notion of
distance or divergence between various segments of the observed distribution.

    For the most up-to-date documentation on the supported metrics, please look at the link [here](lift/src/main/scala/com/linkedin/lift/lib/DivergenceUtils.scala), and look for the
    `computeDatasetDistanceMetrics` method as the starting point. The following metrics are supported for training data:

    1. **Demographic Parity:** It measures the difference between the conditional expected value of the prediction (given one protected attribute value) and the conditional expected value of the prediction (given the other protected attribute value). This is measured for all pairs of protected attribute values.

        ![DP_{(g_1, g_2)} = E\\[Y(X)|G=g_1\\] - E\\[Y(X)|G=g_2\\] = P(Y(X)=1|G=g_1) - P(Y(X)=1|G=g_2)](https://render.githubusercontent.com/render/math?math=DP_%7B(g_1%2C%20g_2)%7D%20%3D%20E%5C%5BY(X)%7CG%3Dg_1%5C%5D%20-%20E%5C%5BY(X)%7CG%3Dg_2%5C%5D%20%3D%20P(Y(X)%3D1%7CG%3Dg_1)%20-%20P(Y(X)%3D1%7CG%3Dg_2))

3. **Aggregate Metrics:** These metrics are useful to obtain higher level (or second order) notions of inequality, when comparing multiple per-protected-attribute-value inequality metrics. For example, these could be used to say if one set of Skews measured is more equally distributed that another set of Skews. These lower-level metrics are called benefit vectors, and the aggregate metrics provide a notion of how uniformly these inequalities are distributed.

    Note that these metrics capture inequalities within the vector. Thus, going by this metric alone is not sufficient. For example, take a benefit vector that captures Demographic Parity differences between (MALE, FEMALE), (FEMALE, UNKNOWN), and (MALE, UNKNOWN). Suppose that the vector for one distribution is (a, 2a, 3a) and the other is (0.5a, 1.5a, 2a). Even though the individual differences are smaller in the second distribution (for each pair of protected attribute values), an aggregate metric will deem it to be more unfair than the former because the differences in the elements of the vector are more drastic than the other (for the first one, the ratio is 1:2:3 while for the second it is 1:3:4). However, the latter has better Demographic Parity. Hence, there may be conflicting notions of fairness being measured, and it is up to the end user to identify which one they would like to focus on.

    For the most up-to-date documentation on the supported metrics, please look at the link [here](lift/src/main/scala/com/linkedin/lift/types/BenefitMap.scala), and look for the `computeMetric` method as the starting point. The following aggregate metrics are available:
    1. **Generalized Entropy Index:** Computes an average of the relative benefits based on some input parameters.
    2. **Atkinsons Index:** A derivative of the Generalized Entropy Index. Used more commonly in the field of economics.
    3. **Theil's L Index:** The Generalized Entropy Index when its parameter is set to 0. It is more sensitive to differences at the lower end of the distribution (the benefit vector values).
    4. **Theil's T Index:** The Generalized Entropy Index when its parameter is set to 1. It is more sensitive to differences at the higher end of the distribution (the benefit vector values).
    5. **Coefficient of Variation:** A derivative of the Generalized Entropy Index. It computes the value of the standard deviation divided by the mean of the benefit vector.
