# Acknowledgements

The computation of AUC and ROC in this library were inspired by the following implementations:
- numpy (3-clause BSD): https://numpy.org/doc/stable/reference/generated/numpy.trapz.html
- spark-mllib (Apache 2.0): https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/evaluation/AreaUnderCurve.scala

The computation of a generalized confusion matrix was inspired by the following implementation:
- AIF360 (Apache 2.0): https://github.com/IBM/AIF360/blob/master/aif360/metrics/classification_metric.py

The permutation test implmented in this library was developed in conjuction with Cyrus DiCiccio, Kinjal Basu and Deepak Agarwal.

The LinkedIn Fairness Toolkit (LiFT) was validated through its deployment as part of different product vertical ML workflows and existing ML
platforms at LinkedIn. Deepak Agarwal, Parvez Ahammad, Stuart Ambler, Romil Bansal, Kinjal Basu, Bee-Chung Chen,
Cyrus DiCiccio, Carlos Faham, Divya Gadde, Priyanka Gariba, Sahin Cem Geyik, Daniel Hewlett, Roshan Lal, Nicole Li,
Heloise Logan, Sofus Macskassy, Varun Mithal, Arashpreet Singh Mor, Tanvi Motwani, Preetam Nandy, Cagri Ozcaglar,
Nitin Panjwani, Igor Perisic, Romer Rosales, Guillaume Saint-Jacques, Badrul Sarwar, Amir Sepehri, Arun Swami,
Ram Swaminathan, Grace Tang, Xin Wang, Ya Xu, and Yang Yang provided insightful feedback and discussions that influenced various aspects of LiFT.
