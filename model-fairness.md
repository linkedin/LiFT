# Model-level Fairness Metrics

At a high level, these metrics require the score of the model and the corresponding
protected attribute value. There are some metrics that also make use of the
corresponding label as well. Now, a model can produce a raw score or a probability.
Our fairness metrics specifically deal with models that output probabilities that
can be treated as ![P(\hat{Y}(X) = 1)](https://render.githubusercontent.com/render/math?math=P(%5Chat%7BY%7D(X)%20%3D%201))
(if the scores are raw scores, we pass it through a sigmoid link function
to interpret it as a probability). If your models
do not output binary prediction probabilities, or these probabilities
are not appropriate to be interpreted as shown, you will need to preprocess
the scores before using the library. This can be done inline, in the Spark
job that computes the model-related fairness metrics.

If your model is being used for binary classification (and not just for its scores), an
optional threshold value can be provided, which will be used to binarize the predictions.
If a threshold value is not specified, the probabilities ![P(\hat{Y}(X) = 1)](https://render.githubusercontent.com/render/math?math=P(%5Chat%7BY%7D(X)%20%3D%201))
are used to compute expected TP, FP, TN and FN counts as needed.

We provide here a list of the various metrics available for measuring fairness of
ML models, as well as a short description of each of them.

1. **Metrics that compare against a given reference distribution:** These metrics involve computing
some measure of distance or divergence from a given reference distribution provided by the user.
The library supports only the `UNIFORM` distribution out of the box (all `score-protectedAttribute`
combinations must have equal number of records), but users may supply their own distribution
(such as an apriori known gender distribution etc.). These metrics are
similar to those computed on the training dataset. The only difference is that we make use
of the predictions/scores instead of the labels, ie., ![\hat{Y}(X)](https://render.githubusercontent.com/render/math?math=%5Chat%7BY%7D(X))
instead of ![Y(X)](https://render.githubusercontent.com/render/math?math=Y(X)).

    For the most up-to-date documentation on the supported metrics, please look at the link [here](lift/src/main/scala/com/linkedin/lift/lib/DivergenceUtils.scala), and look for the
    `computeDistanceMetrics` method as the starting point. The following metrics fall under this
    category:

    1. **Skews:** Computes the logarithm of the ratio of the observed value to the expected value. For example, if we are dealing with score-gender distributions, this metric computes

        ![\log\left(\frac{(0.0, MALE)_{obs}}{(0.0, MALE)_{exp}}\right), \log\left(\frac{(1.0, MALE)_{obs}}{(1.0, MALE)_{exp}}\right), \log\left(\frac{(0.0, FEMALE)_{obs}}{(0.0, FEMALE)_{exp}}\right), \log\left(\frac{(1.0, FEMALE)_{obs}}{(1.0, FEMALE)_{exp}}\right)](https://render.githubusercontent.com/render/math?math=%5Clog%5Cleft(%5Cfrac%7B(0.0%2C%20MALE)_%7Bobs%7D%7D%7B(0.0%2C%20MALE)_%7Bexp%7D%7D%5Cright)%2C%20%5Clog%5Cleft(%5Cfrac%7B(1.0%2C%20MALE)_%7Bobs%7D%7D%7B(1.0%2C%20MALE)_%7Bexp%7D%7D%5Cright)%2C%20%5Clog%5Cleft(%5Cfrac%7B(0.0%2C%20FEMALE)_%7Bobs%7D%7D%7B(0.0%2C%20FEMALE)_%7Bexp%7D%7D%5Cright)%2C%20%5Clog%5Cleft(%5Cfrac%7B(1.0%2C%20FEMALE)_%7Bobs%7D%7D%7B(1.0%2C%20FEMALE)_%7Bexp%7D%7D%5Cright))

    2. **Infinity Norm Distance:** Computes the Chebyshev Distance between the observed and reference distribution. It equals the maximum difference between the two distributions.
    3. **Total Variation Distance:** Computes the Total Variation Distance between the observed and reference distribution. It is equal to half the L1 distance between the two distributions.
    4. **JS Divergence:** The Jensen-Shannon Divergence between the observed and reference distribution. Suppose that the average of these two distributions is given by M. Then, the JS Divergence is the average of the KL Divergences between the observed distribution and M, and the reference distribution and M.
    5. **KL Divergence:** The Kullback-Leibler Divergence between the observed and reference distribution. It is the expectation (over the observed distribution) of the logarithmic differences between the observed and reference distributions. The latter is the Skew we measure above.
    
2. **Metrics computed on the observed distribution only:** These metrics compute some notion of
distance or divergence between various segments of the observed distribution.

    For the most up-to-date documentation on the supported metrics, please look at the link [here](lift/src/main/scala/com/linkedin/lift/lib/DivergenceUtils.scala), and look for the
    `computeDistanceMetrics` method as the starting point. The following metrics fall under this
    category:

    1. **Demographic Parity:** It measures the difference between the conditional expected value of the prediction (given one protected attribute value) and the conditional expected value of the prediction (given the other protected attribute value). This is measured for all pairs of protected attribute values.

        ![DP_{(g_1, g_2)} = E\[\hat{Y}(X)|G=g_1\] - E\[\hat{Y}(X)|G=g_2\] = P(\hat{Y}(X)=1|G=g_1) - P(\hat{Y}(X)=1|G=g_2)](https://render.githubusercontent.com/render/math?math=DP_%7B(g_1%2C%20g_2)%7D%20%3D%20E%5B%5Chat%7BY%7D(X)%7CG%3Dg_1%5D%20-%20E%5B%5Chat%7BY%7D(X)%7CG%3Dg_2%5D%20%3D%20P(%5Chat%7BY%7D(X)%3D1%7CG%3Dg_1)%20-%20P(%5Chat%7BY%7D(X)%3D1%7CG%3Dg_2))

       This metric captures the idea that different protected groups should have similar acceptance rates. While this is desirable in an ideal scenario (and is related to the [80% Labor Law rule](https://en.wikipedia.org/wiki/Disparate_impact), this might not always be true. For example, various socio-economic factors might contribute towards having different acceptance rates for different groups. That is, the difference is not due to the protected group itself, but rather due to other meaningful, but correlated variables. Furthermore, even if we are dealing with a scenario where DP is desirable, it does not deal with model performance at all. We might as well have a second model predict '1' randomly for one group (with a probability equal to the acceptance rate of the other group) to achieve DP. Thus, attempting to optimize for DP directly might not be a good goal, but using it to inform decisions is nevertheless helpful.

    2. **Equalized Odds:** It measures the difference between the conditional expected value of the prediction (given one protected attribute value and its label) and the conditional expected value of the prediction (given the other protected attribute value and its label). This is measured for all pairs of protected attribute values and label.

        ![EO_{(g_1, g_2, y)} = E\[\hat{Y}(X)|Y=y,G=g_1\] - E\[\hat{Y}(X)|Y=y,G=g_2\] = P(\hat{Y}(X)=1|Y=y,G=g_1) - P(\hat{Y}(X)=1|Y=y,G=g_2)](https://render.githubusercontent.com/render/math?math=EO_%7B(g_1%2C%20g_2%2C%20y)%7D%20%3D%20E%5B%5Chat%7BY%7D(X)%7CY%3Dy%2CG%3Dg_1%5D%20-%20E%5B%5Chat%7BY%7D(X)%7CY%3Dy%2CG%3Dg_2%5D%20%3D%20P(%5Chat%7BY%7D(X)%3D1%7CY%3Dy%2CG%3Dg_1)%20-%20P(%5Chat%7BY%7D(X)%3D1%7CY%3Dy%2CG%3Dg_2))

3. **Statistical Tests for Fairness:** This deals with comparing a given model performance metric between two different protected groups. For example, comparing the AUC for men vs AUC for women. We need to be able to say if this difference is statistically significant, and we also need it to be metric-agnostic. We achieve this using Permutation Testing. Since this is a non-parametric statistical test, it can be slow, so users can control the sample size and the number of trials to run. The test provides a p-value and a measure of standard error (for the p-value) as well.

    We support AUC, Precision, Recall, TNR, FNR and FPR out-of-the-box (the full list can be found by visiting `StatsUtils.scala` and looking at the `getMetricFn` method), and also support any user-defined custom metrics (it needs to extend [CustomMetric.scala](lift/src/main/scala/com/linkedin/lift/types/CustomMetric.scala)). More details about the test itself can be found in the `permutationTest` method defined [here](lift/src/main/scala/com/linkedin/lift/lib/PermutationTestUtils.scala). To cite this work, please refer to the 'Citations' section of the [README](README.md).
    
4. **Aggregate Metrics:** These metrics are useful to obtain higher level (or second order) notions of inequality, when comparing multiple per-protected-attribute-value inequality metrics. For example, these could be used to say if one set of Skews measured is more equally distributed that another set of Skews. These lower-level metrics are called benefit vectors, and the aggregate metrics provide a notion of how uniformly these inequalities are distributed.

    Note that these metrics capture inequalities within the vector. Thus, going by this metric alone is not sufficient. For example, take a benefit vector that captures Demographic Parity differences between (MALE, FEMALE), (FEMALE, UNKNOWN), and (MALE, UNKNOWN). Suppose that the vector for one distribution is (a, 2a, 3a) and the other is (0.5a, 1.5a, 2a). Even though the individual differences are smaller in the second distribution (for each pair of protected attribute values), an aggregate metric will deem it to be more unfair than the former because the differences in the elements of the vector are more drastic than the other (for the first one, the ratio is 1:2:3 while for the second it is 1:3:4). However, the latter has better Demographic Parity. Hence, there may be conflicting notions of fairness being measured, and it is up to the end user to identify which one they would like to focus on.

    We divide these into two: `distanceBenefitMetrics` and `performanceBenefitMetrics`. The former computes distance and divergence metrics (mentioned in 1 and 2) and uses these as the benefit vectors for aggregate metrics computation. The latter uses model performance metrics (such as AUC, TPR, FPR for different protected groups) as the benefit vector for aggregate metrics computation. There is no difference in the aggregate computation itself; this distinction is used by LiFT to just be more specific about what needs to be computed. The aggregate metrics can be computed for performance metrics supported out-of-the-box, as well as user-defined custom ones, as mentioned in 3.

    For the most up-to-date documentation on the supported metrics, please look at the link [here](lift/src/main/scala/com/linkedin/lift/types/BenefitMap.scala), and look for the `computeMetric` method as the starting point. The following aggregate metrics are available:
    1. **Generalized Entropy Index:** Computes an average of the relative benefits based on some input parameters.
    2. **Atkinsons Index:** A derivative of the Generalized Entropy Index. Used more commonly in the field of economics.
    3. **Theil's L Index:** The Generalized Entropy Index when its parameter is set to 0. It is more sensitive to differences at the lower end of the distribution (the benefit vector values).
    4. **Theil's T Index:** The Generalized Entropy Index when its parameter is set to 1. It is more sensitive to differences at the higher end of the distribution (the benefit vector values).
    5. **Coefficient of Variation:** A derivative of the Generalized Entropy Index. It computes the value of the standard deviation divided by the mean of the benefit vector.

