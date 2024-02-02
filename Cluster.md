---
name: Cluster
topic: Cluster Analysis & Finite Mixture Models
maintainer: Friedrich Leisch, Bettina Gruen
email: Bettina.Gruen@R-project.org
version: 2024-02-02
source: https://github.com/cran-task-views/Cluster/
---

This CRAN Task View contains a list of packages that can be used for
finding groups in data and modeling unobserved cross-sectional
heterogeneity. Many packages provide functionality for more than one
of the topics listed below, the section headings are mainly meant as
quick starting points rather than as an ultimate
categorization. Except for packages stats and `r pkg("cluster", priority = "core")`
(which essentially ship with base R and hence are part of every R
installation), each package is listed only once.

Most of the packages listed in this view, but not all, are distributed
under the GPL. Please have a look at the DESCRIPTION file of each
package to check under which license it is distributed.

### Hierarchical Clustering:

-   Functions `hclust()` from package stats and `agnes()` from
    `r pkg("cluster")` are the primary
    functions for agglomerative hierarchical clustering, function
    `diana()` can be used for divisive hierarchical clustering. Faster
    alternatives to `hclust()` are provided by the packages
    `r pkg("fastcluster")` and
    `r pkg("flashClust")`.
-   Function `dendrogram()` from package stats and associated 
    methods can be used for improved visualization for cluster dendrograms.
-   Package `r pkg("dendextend")` provides functions for
    easy visualization (coloring labels and branches, etc.),
    manipulation (rotating, pruning, etc.) and comparison of dendrograms
    (tangelgrams with heuristics for optimal branch rotations, and tree
    correlation measures with bootstrap and permutation tests for
    significance).
-   Package `r pkg("dynamicTreeCut")` contains methods for
    detection of clusters in hierarchical clustering dendrograms.
-   Package `r pkg("genieclust")` implements a fast
    hierarchical clustering algorithm with a linkage criterion which is
    a variant of the single linkage method combining it with the Gini
    inequality measure to robustify the linkage method while retaining
    computational efficiency to allow for the use of larger data sets.
-   Package `r pkg("hclust1d")` provides univariate agglomerative
    hierarchical clustering for a comprehensive choice of linkage
    functions based on an $O(n \log(n))$ algorithm implemented in C++.
-   Package `r pkg("idendr0")` allows to interactively
    explore hierarchical clustering dendrograms and the clustered data.
    The data can be visualized (and interacted with) in a built-in heat
    map, but also in GGobi dynamic interactive graphics (provided by
    rggobi), or base R plots.
-   Package `r pkg("mdendro")` provides an alternative
    implementation of agglomerative hierarchical clustering. The package
    natively handles similarity matrices, calculates variable-group
    dendrograms, which solve the non-uniqueness problem that arises when
    there are ties in the data, and calculates five descriptors for the
    final dendrogram: cophenetic correlation coefficient, space
    distortion ratio, agglomerative coefficient, chaining coefficient,
    and tree balance.
-   Package `r pkg("protoclust")` implements a form of
    hierarchical clustering that associates a prototypical element with
    each interior node of the dendrogram. Using the package's `plot()`
    function, one can produce dendrograms that are prototype-labeled and
    are therefore easier to interpret.
-   Package `r pkg("pvclust")` assesses the uncertainty in hierarchical
    cluster analysis. It provides approximately unbiased p-values as
    well as bootstrap p-values.

### Partitioning Clustering:

-   Function `kmeans()` from package stats provides several algorithms
    for computing partitions with respect to Euclidean distance.
-   Function `pam()` from package `r pkg("cluster")`
    implements partitioning around medoids and can work with arbitrary
    distances. Function `clara()` is a wrapper to `pam()` for larger
    data sets. Silhouette plots and spanning ellipses can be used for
    visualization.
-   Package `r pkg("apcluster")` implements Frey's and
    Dueck's Affinity Propagation clustering. The algorithms in the
    package are analogous to the Matlab code published by Frey and
    Dueck.
-   Package `r pkg("ClusterR")` implements k-means,
    mini-batch-kmeans, k-medoids, affinity propagation clustering and
    Gaussian mixture models with the option to plot, validate, predict
    (new data) and estimate the optimal number of clusters. The package
    takes advantage of RcppArmadillo to speed up the computationally
    intensive parts of the functions.
-   Package `r pkg("clusterSim")` allows to search for the
    optimal clustering procedure for a given dataset.
-   Package `r pkg("clustMixType")` implements Huang's
    k-prototypes extension of k-means for mixed type data.
-   Package `r pkg("evclust")` implements various clustering
    algorithms that produce a credal partition, i.e., a set of
    Dempster-Shafer mass functions representing the membership of
    objects to clusters.
-   Package `r pkg("flexclust", priority = "core")` provides
    k-centroid cluster algorithms for arbitrary distance measures, hard
    competitive learning, neural gas and QT clustering. Neighborhood
    graphs and image plots of partitions are available for
    visualization. Some of this functionality is also provided by
    package `r pkg("cclust")`.
-   Package `r pkg("kernlab")` provides a weighted kernel
    version of the k-means algorithm by `kkmeans` and spectral
    clustering by `specc`.
-   Package `r pkg("kml")` provides k-means clustering
    specifically for longitudinal (joint) data.
-   Package `r pkg("QuClu")` provides high-dimensional clustering
    with potentially skew cluster-wise distributions representing
    clusters by quantiles.
-   Package `r pkg("skmeans")` allows spherical k-Means
    Clustering, i.e. k-means clustering with cosine similarity. It
    features several methods, including a genetic and a simple
    fixed-point algorithm and an interface to the CLUTO vcluster program
    for clustering high-dimensional datasets.
-   Package `r pkg("Spectrum")` implements a self-tuning
    spectral clustering method for single or multi-view data and uses
    either the eigengap or multimodality gap heuristics to determine the
    number of clusters. The method is sufficiently flexible to cluster a
    wide range of Gaussian and non-Gaussian structures with automatic
    selection of K.
-   Package `r pkg("tclust")` allows for trimmed k-means
    clustering. In addition using this package other covariance
    structures can also be specified for the clusters.

### Model-Based Clustering:

-   ML estimation:
    -   For semi- or partially supervised problems, where for a part of
        the observations labels are given with certainty or with some
        probability, package `r pkg("bgmm")` provides
        belief-based and soft-label mixture modeling for mixtures of
        Gaussians with the EM algorithm.
    -   Package `r pkg("EMCluster")` provides EM algorithms and
        several efficient initialization methods for model-based
        clustering of finite mixture Gaussian distribution with
        unstructured dispersion in unsupervised as well as
        semi-supervised learning situation.
    -   Package `r pkg("funFEM")` provides model-based functional data
        analysis by implementing the funFEM algorithm which
        allows to cluster time series or, more generally, functional
        data. It is based on a discriminative functional mixture model
        which allows the clustering of the data in a unique and
        discriminative functional subspace. This model presents the
        advantage to be parsimonious and can therefore handle long
        time series.
    -   Package `r pkg("GLDEX")` fits mixtures of
        generalized lambda distributions and for grouped conditional
        data package `r pkg("mixdist")` can be used.
    -   Package `r pkg("GMCM")` fits Gaussian mixture copula
        models for unsupervised clustering and meta-analysis.
    -   Package `r pkg("HDclassif")` provides function
        `hddc` to fit Gaussian mixture model to high-dimensional data
        where it is assumed that the data lives in a lower dimension
        than the original space.
    -   Package `r pkg("teigen")` allows to fit multivariate
        t-distribution mixture models (with eigen-decomposed covariance
        structure) from a clustering or classification point of view.
    -   Package `r pkg("mclust", priority = "core")` fits
        mixtures of Gaussians using the EM algorithm. It allows fine
        control of volume and shape of covariance matrices and
        agglomerative hierarchical clustering based on maximum
        likelihood. It provides comprehensive strategies using
        hierarchical clustering, EM and the Bayesian Information
        Criterion (BIC) for clustering, density estimation, and
        discriminant analysis. Package
        `r pkg("Rmixmod", priority = "core")` provides tools
        for fitting mixture models of multivariate Gaussian or
        multinomial components to a given data set with either a
        clustering, a density estimation or a discriminant analysis
        point of view. Package `r pkg("mclust")` as well as
        packages `r pkg("mixture")` and
        `r pkg("Rmixmod")` provide all 14 possible
        variance-covariance structures based on the eigenvalue
        decomposition.
    -   Package `r pkg("MetabolAnalyze")` fits mixtures of
        probabilistic principal component analysis with the EM
        algorithm.
    -   For grouped conditional data package
        `r pkg("mixdist")` can be used.
    -   Package `r pkg("MixAll")` provides EM estimation of
        diagonal Gaussian, gamma, Poisson and categorical mixtures
        combined based on the conditional independence assumption using
        different EM variants and allowing for missing observations. The
        package accesses the clustering part of the Statistical ToolKit
        [STK++](https://www.stkpp.org/).
    -   Package `r pkg("mixR")` performs maximum likelihood
        estimation of finite mixture models for raw or binned data for
        families including Normal, Weibull, Gamma and Lognormal using
        the EM algorithm, together with the Newton-Raphson algorithm or
        the bisection method when necessary. The package also provides
        information criteria or the bootstrap likelihood ratio test for
        model selection and the model fitting process is accelerated
        using package Rcpp.
    -   Package `r pkg("mixtools")` provides fitting with the EM
        algorithm for parametric and non-parametric (multivariate)
        mixtures. Parametric mixtures include mixtures of multinomials,
        multivariate normals, normals with repeated measures, Poisson
        regressions and Gaussian regressions (with random effects).
        Non-parametric mixtures include the univariate semi-parametric
        case where symmetry is imposed for identifiability and
        multivariate non-parametric mixtures with conditional
        independent assumption. In addition fitting mixtures of Gaussian
        regressions with the Metropolis-Hastings algorithm is available.
    -   Fitting finite mixtures of uni- and multivariate scale mixtures
        of skew-normal distributions with the EM algorithm is provided
        by package `r pkg("mixsmsn")`.
    -   Package `r pkg("MoEClust")` fits parsimonious finite
        multivariate Gaussian mixtures of experts models via the EM
        algorithm. Covariates may influence the mixing proportions
        and/or component densities and all 14 constrained covariance
        parameterizations from package `r pkg("mclust")` are
        implemented.
    -   Package `r pkg("movMF")` fits finite mixtures of von
        Mises-Fisher distributions with the EM algorithm.
    -   Package `r pkg("otrimle")` performs robust cluster analysis
        allowing for outliers and noise that cannot be fitted by any
        cluster. The data are modeled by a mixture of Gaussian
        distributions and a noise component, which is an improper
        uniform distribution covering the whole Euclidean space.
    -   Package `r pkg("prabclus")` clusters a presence-absence
        matrix object by calculating an MDS from the distances, and
        applying maximum likelihood Gaussian mixtures clustering to the
        MDS points.
    -   Package `r pkg("psychomix")` estimates mixtures of
        the dichotomous Rasch model (via conditional ML) and the
        Bradley-Terry model.
    -   Package `r pkg("rebmix")` implements the REBMIX
        algorithm to fit mixtures of conditionally independent normal,
        lognormal, Weibull, gamma, binomial, Poisson, Dirac or von Mises
        component densities as well as mixtures of multivariate normal
        component densities with unrestricted variance-covariance
        matrices.
    -   Package `r pkg("RMixtComp")` performs clustering using mixture
        models with heterogeneous data and partially missing data. The
        mixture models are fitted using a SEM algorithm and the
        package includes 8 models for real, categorical, counting,
        functional and ranking data.
    -   Package `r pkg("stepmixr")` allows model-based clustering and
        generalized mixture modeling (latent class/profile analysis) of
        continuous and categorical data. In addition, the `r pkg("stepmixr")`
        package provides multiple stepwise EM estimation methods (p. ex., 2-step,
        BCH, and ML) for analyzing covariates and/or distal outcomes, handles
        missing values through FIML, and allows inference in semi-supervised
        and unsupervised settings with non-parametric bootstrapping.
-   Bayesian estimation:
    -   Bayesian estimation of finite mixtures of multivariate Gaussians
        is possible using package `r pkg("bayesm")`. The
        package provides functionality for sampling from such a mixture
        as well as estimating the model using Gibbs sampling. Additional
        functionality for analyzing the MCMC chains is available for
        averaging the moments over MCMC draws, for determining the
        marginal densities, for clustering observations and for plotting
        the uni- and bivariate marginal densities.
    -   Package `r pkg("bayesmix")` provides Bayesian
        estimation using JAGS.
    -   Package `r pkg("bmixture")` provides Bayesian
        estimation of finite mixtures of univariate Gamma and normal
        distributions.
    -   Package `r pkg("GSM")` fits mixtures of gamma
        distributions.
    -   Package `r pkg("IMIFA")` fits Infinite Mixtures of
        Infinite Factor Analyzers and a flexible suite of related models
        for clustering high-dimensional data. The number of clusters
        and/or number of cluster-specific latent factors can be
        non-parametrically inferred, without recourse to model selection
        criteria.
    -   Package `r pkg("mcclust")` implements methods for
        processing a sample of (hard) clusterings, e.g. the MCMC output
        of a Bayesian clustering model. Among them are methods that find
        a single best clustering to represent the sample, which are
        based on the posterior similarity matrix or a relabeling
        algorithm.
    -   Package `r pkg("mixAK")` contains a mixture of
        statistical methods including the MCMC methods to analyze normal
        mixtures with possibly censored data.
    -   Package `r pkg("NPflow")` fits Dirichlet process
        mixtures of multivariate normal, skew normal or skew
        t-distributions. The package was developed oriented towards
        flow-cytometry data preprocessing applications.
    -   Package `r pkg("PReMiuM")` is a package for profile
        regression, which is a Dirichlet process Bayesian clustering
        where the response is linked non-parametrically to the covariate
        profile.
    -   Package `r pkg("rjags")` provides an interface to
        the JAGS MCMC library which includes a module for mixture
        modelling.
-   Other estimation methods:
    -   Package `r pkg("AdMit")` allows to fit an adaptive
        mixture of Student-t distributions to approximate a target
        density through its kernel function.

### Other Cluster Algorithms and Clustering Suites:

-   Package `r pkg("ADPclust")` allows to cluster high
    dimensional data based on a two dimensional decision plot. This
    density-distance plot plots for each data point the local density
    against the shortest distance to all observations with a higher
    local density value. The cluster centroids of this non-iterative
    procedure can be selected using an interactive or automatic
    selection mode.
-   Package `r pkg("amap")` provides alternative
    implementations of k-means and agglomerative hierarchical
    clustering.
-   Package `r pkg("biclust")` provides several algorithms
    to find biclusters in two-dimensional data.
-   Package `r pkg("cba")` implements clustering techniques
    for business analytics like "rock" and "proximus".
-   Package `r pkg("clue")` implements ensemble methods for
    both hierarchical and partitioning cluster methods.
-   Package `r pkg("CoClust")` implements a cluster
    algorithm that is based on copula functions and therefore allows to
    group observations according to the multivariate dependence
    structure of the generating process without any assumptions on the
    margins.
-   Package `r pkg("compHclust")` provides complimentary
    hierarchical clustering which was especially designed for microarray
    data to uncover structures present in the data that arise from
    'weak' genes.
-   Package `r pkg("DatabionicSwarm")` implements a swarm
    system called Databionic swarm (DBS) for self-organized clustering.
    This method is able to adapt itself to structures of
    high-dimensional data such as natural clusters characterized by
    distance and/or density based structures in the data space.
-   Package `r pkg("dbscan")` provides a fast
    reimplementation of the DBSCAN (density-based spatial clustering of
    applications with noise) algorithm using a kd-tree.
-   Fuzzy clustering and bagged clustering are available in package
    `r pkg("e1071")`. Further and more extensive tools for
    fuzzy clustering are available in package
    `r pkg("fclust")`.
-   Package `r pkg("FCPS")` provides many conventional
    clustering algorithms with consistent input and output, several
    statistical approaches for the estimation of the number of clusters
    as well as the mirrored density plot (MD-plot) of clusterability and
    offers a variety of clustering challenges any algorithm should be
    able to handle when facing real world data.
-   The `r bioc("hopach")` algorithm is a hybrid between
    hierarchical methods and PAM and builds a tree by recursively
    partitioning a data set.
-   For graphs and networks model-based clustering approaches are
    implemented in `r pkg("latentnet")`.
-   Package `r pkg("ORIClust")` provides order-restricted
    information-based clustering, a cluster algorithm which has
    specifically been developed for bioinformatics applications.
-   Package `r pkg("pdfCluster")` provides tools to perform
    cluster analysis via kernel density estimation. Clusters are
    associated to the maximally connected components with estimated
    density above a threshold. In addition a tree structure associated
    with the connected components is obtained.
-   Package `r pkg("prcr")` implements the 2-step cluster
    analysis where first hierarchical clustering is performed to
    determine the initial partition for the subsequent k-means
    clustering procedure.
-   Package `r pkg("ProjectionBasedClustering")` implements
    projection-based clustering (PBC) for high-dimensional datasets in
    which clusters are formed by both distance and density structures
    (DDS).
-   Package `r pkg("randomLCA")` provides the fitting of
    latent class models which optionally also include a random effect.
    Package `r pkg("poLCA")` allows for polytomous variable
    latent class analysis and regression.
    `r pkg("BayesLCA")` allows to fit Bayesian LCA models
    employing the EM algorithm, Gibbs sampling or variational Bayes
    methods.
-   Package `r pkg("RPMM")` fits recursively partitioned
    mixture models for Beta and Gaussian Mixtures. This is a model-based
    clustering algorithm that returns a hierarchy of classes, similar to
    hierarchical clustering, but also similar to finite mixture models.
-   Self-organizing maps are available in package
    `r pkg("som")`.

### Cluster-wise Regression:

-   Package `r pkg("crimCV")` fits finite mixtures of
    zero-inflated Poisson models for longitudinal data with time as
    covariate.
-   Multigroup mixtures of latent Markov models on mixed categorical and
    continuous data (including time series) can be fitted using
    `r pkg("depmix")` or `r pkg("depmixS4")`.
    The parameters are optimized using a general purpose optimization
    routine given linear and nonlinear constraints on the parameters.
-   Package `r pkg("flexmix", priority = "core")` implements
    an user-extensible framework for EM-estimation of mixtures of
    regression models, including mixtures of (generalized) linear
    models.
-   Package `r pkg("fpc")` provides fixed-point methods both
    for model-based clustering and linear regression. A collection of
    asymmetric projection methods can be used to plot various aspects of
    a clustering.
-   Package `r pkg("lcmm")` fits a latent class linear mixed
    model which is also known as growth mixture model or heterogeneous
    linear mixed model using a maximum likelihood method.
-   Package `r pkg("mixreg")` fits mixtures of one-variable
    regressions and provides the bootstrap test for the number of
    components.
-   Package `r pkg("mixPHM")` fits mixtures of proportional hazard
    models with the EM algorithm.

### Additional Functionality:

-   Package `r pkg("clusterGeneration")` contains functions
    for generating random clusters and random covariance/correlation
    matrices, calculating a separation index (data and population
    version) for pairs of clusters or cluster distributions, and 1-D and
    2-D projection plots to visualize clusters. Alternatively
    `r pkg("MixSim")` generates a finite mixture model with
    Gaussian components for prespecified levels of maximum and/or
    average overlaps. This model can be used to simulate data for
    studying the performance of cluster algorithms.
-   Package `r pkg("clusterCrit")` computes various
    clustering validation or quality criteria and partition comparison
    indices.
-   For cluster validation package `r pkg("clusterRepro")`
    tests the reproducibility of a cluster. Package
    `r pkg("clv")` contains popular internal and external
    cluster validation methods ready to use for most of the outputs
    produced by functions from package `r pkg("cluster")`
    and `r pkg("clValid")` calculates several stability
    measures.
-   Package `r pkg("clustvarsel")` provides variable
    selection for Gaussian model-based clustering. Variable selection
    for latent class analysis for clustering multivariate categorical
    data is implemented in package `r pkg("LCAvarsel")`.
    Package `r pkg("VarSelLCM")` provides variable selection
    for model-based clustering of continuous, count, categorical or
    mixed-type data with missing values where the models used impose a
    conditional independence assumption given group membership.
-   Package `r pkg("factoextra")` provides some easy-to-use
    functions to extract and visualize the output of multivariate data
    analyses in general including also heuristic and model-based cluster
    analysis. The package also contains functions for simplifying some
    cluster analysis steps and uses ggplot2-based visualization.
-   Functionality to compare the similarity between two cluster
    solutions is provided by `cluster.stats()` in package
    `r pkg("fpc")`.
-   The stability of k-centroid clustering solutions fitted using
    functions from package `r pkg("flexclust")` can also be
    validated via `bootFlexclust()` using bootstrap methods.
-   Package `r pkg("MOCCA")` provides methods to analyze
    cluster alternatives based on multi-objective optimization of
    cluster validation indices.
-   Package `r pkg("NbClust")` implements 30 different
    indices which evaluate the cluster structure and should help to
    determine on a suitable number of clusters.
-   Mixtures of univariate normal distributions can be printed and
    plotted using package `r pkg("nor1mix")`.
-   Package `r pkg("seriation")` provides `dissplot()` for
    visualizing dissimilarity matrices using seriation and matrix
    shading. This also allows to inspect cluster quality by restricting
    objects belonging to the same cluster to be displayed in consecutive
    order.
-   Package `r pkg("sigclust")` provides a statistical
    method for testing the significance of clustering results.
-   Package `r pkg("treeClust")` calculates dissimilarities
    between data points based on their leaf memberships in regression or
    classification trees for each variable. It also performs the cluster
    analysis using the resulting dissimilarity matrix with available
    heuristic clustering algorithms in R.

