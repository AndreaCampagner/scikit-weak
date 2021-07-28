# scikit-weak (scikit-weakly-supervised)
 A package featuring utilities and algorithms for weakly supervised ML.
 Should be (more-or-less) compatible with scikit-learn!
 It collects original algorithms and methods developed at the MUDI lab (DISCo dept., University of Milano-Bicocca, Milan, Italy),
 as well as some algorithms available in the literature.

 ## How to install
 You can install the library using the command:

 ```
 pip install scikit-weak
 ```
 
 ### Dependencies:
 numpy, scipy, scikit-learn, pandas

 ## Documentation
 The documentation is generated using Sphinx (https://www.sphinx-doc.org/). 
 If you download the source code from this repository you can generate the documentation in html format by typing: 
 ```
 sphinx-build -b html docs/source docs/build/html
 ```
 in the main folder of the project.
 
 ## References:

 [1] Campagner, A., Ciucci, D., Hullermeier, E. (2021). Rough set-based feature selection for weakly labeled data. International Journal of Approximate Reasoning, 136, 150-167. https://doi.org/10.1016/j.ijar.2021.06.005.

 [2] Campagner, A., Ciucci, D., Svensson, C. M., Figge, M. T., & Cabitza, F. (2021). Ground truthing from multi-rater labeling with three-way decision and possibility theory. Information Sciences, 545, 771-790. https://doi.org/10.1016/j.ins.2020.09.049  

 [3] Campagner, A., Ciucci, D., & Hüllermeier, E. (2020). Feature Reduction in Superset Learning Using Rough Sets and Evidence Theory. In International Conference on Information Processing and Management of Uncertainty in Knowledge-Based Systems (pp. 471-484). Springer, Cham. https://doi.org/10.1007/978-3-030-50146-4_35

