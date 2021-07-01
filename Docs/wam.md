

# WAM: A Weight Array Model for Prediction of Eukaryotic Genetic Splice Sites

## Abstract

Splice sites are a vitally important gene sequence pattern amongst the functional sites inside a eukaryotic gene. However, splice site prediction does not come easy thanks to the extreme complexity of human genome. In this paper, an optimized frequency-based method to predict eukaryotic genetic splice site patterns with the Weight Array Model (WAM) is proposed, which is a feasible lightweight computational approach for gene functional site finding, in order to deal with splice site predictions. We prove its accuracy and high efficiency by comparison studies on the renowned Kulp \& Reese human genome dataset, during which we achieve excellent results on several different metrics. The source code is available on GitHub and can be obtained from https://github.com/Newiz430/SplicePredictor. 

## 1 Introduction

Gene finding by computational methodologies, the foundation for all further investigation in functional genomics, has attracted considerable research attention since the 20^th^ century \cite{burge1997prediction}. With the thriving of functional genomics after the completion of Human Genome Project (HGP), functions of the elements of eukaryotic gene sequences were beginning to surface. Researchers came to realize that DNA sequences, other than genes, contain a huge amount of information, most of which is correlated with the structural features of nucleic acids and in general determines the DNA - protein, or DNA - RNA interactions. Such information is mostly provided by a variety of functional sites (i.e. sequence motif). The splice sites are a vitally important eukaryotic gene sequence pattern amongst all these sites. As terminal points of RNA splicing, splice sites label the junction of transcript exons and introns, assisting biologists in identifying and positioning the coding sequence within a gene. Splicing, itself, also influences the structure and function of genes, which makes genes more "modular", allowing new combinations of exons to be created during evolution. Furthermore, new exons can be inserted into old introns, creating new proteins without disrupting the function of the old gene \cite{clancy2008rna}. Hence the discovery and prediction of splice sites are of great significance for genetic selective expression research. 

A splice site locates in the edge of an intron, including a donor site (5' end of the intron) and an acceptor site (3' end of the intron). As a typical sequence motif, the donor site includes an almost invariant sequence GU at the 5' end of the intron, within a larger, less highly conserved region. The splice acceptor site at the 3' end terminates the intron with an almost invariant AG sequence. \cite{black2003mechanisms} Some sections of the intron foretell the positions of these two sites. For example, a fragment of sequence upstream of the acceptor consisting of cytosines and thymines, which is called a polypyrimidine tract \cite{lodish2008molecular}. Fig. 1 shows the base distributions adjacent of the splice donor sites and acceptor sites. 

<center class = "half" > 
    <img src = "D:\Project\SplicePredictor\Pics\donor\donor_logo.png" width = "500">
    <img src = "D:\Project\SplicePredictor\Pics\acceptor\acceptor_logo.png" width = "500">
</center>

**Figure 1** Base distribution represented as sequence logos \cite{schneider1990sequence} for eukaryotic gene splice sites (5 upstream sites, 2 conservative sites \& 5 downstream sites): (a) donor site; (b) acceptor site. The most conserved sites is revealed within logo pictures: for donor sites, GT (6, 7); acceptor sites, AG (6, 7). Despite these two, distribution of adjacent sites (5, 8, 9, 10 for the donor, 5 for the acceptor) appear to have some consistency, too. The polypyrimidine tract can also be observed upstream of the acceptor site (1, 2, 3). The information content at a certain point $R_i$ is given by $R_i = \log_24 - (H_i + e_n)$ where $H_i = -\sum\limits_{i=1}^4{P_i(\log_2{P_i})}$ is the Shannon entropy \cite{shannon1948mathematical} , using bit as the basic unit. Higher the bits, higher the relative frequency of that base \cite{schneider1986information}. 

As a matter of fact, accurate prediction does not come easy thanks to the extreme complexity of human genome. On one hand, the number and length of exons and introns in a eukaryotic gene exhibit great uncertainty. One eukaryotic gene contains 5.48 exons with 30 - 36 bps long on average. While the longest exon in the human genome is 11555 bp long, several exons have been found to be only 2 bp long \cite{sakharkar2004distributions}. On the other, the existence of alternate splicing make it harder to predict \cite{black2003mechanisms}. In this paper, we apply a feasible lightweight computational approach for gene functional site finding to predict eukaryotic gene splice sites, and prove its accuracy and high efficiency. 

### 1.1 Related Work

Several typical computational methods that attempt to predict eukaryotic splice sites from unknown gene sequences (i.e. *ab initio* prediction) have been proposed previously. 

Frequency-based methods count the nucleotide frequencies of each site via multiple sequence alignment, etc. and work out the log-odds ratio to compare and find conservative sections in the alignment results. Rodger, et al. (1983) \cite{staden1984computer} proposed a computational model using a weight matrix to represent each type of recognition sequence. A weight matrix is a two dimensional array of values that represent the score for finding each of the possible sequence characters at each position in the target signal. The Weight Matrix Model (WMM) now becomes deprecated owing to its poor accuracy and its independence assumption, that is, WMM only takes point-wise base distribution into consideration, regardless of the potential dependence between adjacent points which is more conformable to the realistic situations. 

Bayesian methods are ones that consider long range dependency among nucleotide sequences. Chen, et al. (2005) \cite{chen2005prediction} proposed a method using dependency graphs and its expanded form - Bayesian Network to fully capture the intrinsic interdependency between base positions in a splice site. The entire model, including the DAG structures and conditional probabilities, is learned ab initio from a training set. However, with astronomical computation complexity and model training difficulty, the performance of Bayesian models is not in keeping with them. 

Supervised learning methods learn a model from existing training set which is able to identify the most effective pattern automatically. Duan, et al. (2008) \cite{duan2008position} developed the support vector machine (SVM) for position-specific residue preference feature prediction which determines the second structure of double helices. Ryen et al. (2008) \cite{ryen2008splice} introduced the artificial neural network (ANN) in this area and trained the model with backpropagation, which can make predictions without prior knowledge of any sensor signals. Accurate and efficient learning approaches they are, supervised learning methods are heavily dependent on the mass and quality of training sets. Models may not be improved and a computational resource waste may happen when an unbalanced dataset or one with too many noises is provided. For SVMs, kernel function selection is a tricky problem, and neural networks acquire a suitable framework and initial hyperparameters. 

### 1.2 Contributions

In this work, we propose an optimized frequency-based method to predict splice site patterns with the weight array model. At bottom, a weight array method continues to extract splice signals, count the frequencies of nucleotides and fill the matrices, identical with WMM. What we can use to distinguish WAM from WMM is that WAM takes into account the correspondence between current position and an adjoining position, which we certify conducive to promote accuracy of splice site prediction. Our contributions are listed as follows. 

- We implemented the weight array model by Python using the given KR set and estimated its performance on the BG set, referring to the existing experiment by Zhang et al. (1993) \cite{zhang1993weight}
- We did a comparison study between WMM and WAM model to prove the superiority of our model. 
- We applied our model on the prediction of both donor splice sites and acceptor splice sites. 

## 2 Methods

Our method is illustrated in Fig. 2, which mainly contains two parts. The statistics of base pair distribution is carried out at the "training" step, and sequences of the testing set are scored by probability matrices at the "predicting" step. See below for more detailed presentation. 

![WAM](D:\Project\SplicePredictor\Pics\WAM.svg)

**Figure 2** Overall architecture of the WAM splice predictor. We use the training set to create two matrices by these steps: extracting positive site signals \& randomly choosing negative signals, encoding \& counting the bases for each position, calculating distribution probabilities, filling the matrices and saving for future predictions. We use P matrices to predict unknown signals by these steps: extracting testing sequences of same length by window-sliding, scoring the sequences position-wise with obtained probabilities, calculating the binary log-odds scores, and comparing them with a given threshold to make final judgment. (Framework picture made by hand)

### 2.1 Hypothesis

WAM is suitable for predicting work only when the two assumptions below stand. Firstly, we assume conservation around the functional splice sites through the entire experiment. Additionally, we consider the intrinsic interdependency only exists between adjacent sites. As for data, we assume that everything about the splice site pattern remained unknown (including the obligatory sites GT / AG) until we dug them out, in order to guarantee the generality of our model, since our aim is to make it possible to branch out to other unidentified functional sites. 

### 2.2 Data Extraction

Dash et al. (2001) \cite{dash2001modeling} found in predicting splice sites by a Bayesian network that it achieves better performance when both of the upstream and downstream feature lengths are greater than 15. With a view to simplifying model and decreasing the computation, we choose 5 upstream sites and 7 downstream sites of intron / exon junctions to form 12 nt long signal sequences from the primary training set. We abandon sequences containing ambiguous bases, whose correspondence with the splice sites we consider inapparent. The training set provides 2,381 donor signals and 2,381 acceptor signals. As for negative samples, Chen, et al. \cite{chen2005prediction} used pseudo splice sites as false data, extracted by searching for negative sample sequences with $P_{+1}P_{+2} = \text{GU / AG}$ whereas, according to the splice site hypothesis above, We randomly selected about 5,000 sites in sections which do not intersect with all donor and acceptor sites, and combined with positive ones to get actual training dataset, the positive-negative ratio of which is about 1:2. Additionally, we export sequences with the same length by window sliding from the primal testing set and build the actual testing set in a positive-negative ratio of 1:20. 

### 2.3 Construction of Probability Matrices

For convenient computing, we encode the extracted sequences. We use 0, 1, 2, and 3 to represent bases A, C, G, T which indicates the position in the P matrices of each base. For positive samples with the length of $\lambda$, we create a $\lambda$ * 4 * 4 probability matrix $P^+$, in which each position  $(N_j, N_{j+1})$ of the $i$th 4 * 4 submatrix stores the conditional probability $P^+[i, N_j, N_{j+1}]$ which denotes the probability of $N_j$ at the current position $j$ if the nucleotide at position $j+1$ is $N_{j+1}$. Each probability is calculated by counting the frequencies of every situations $\text{freq}(i, N_j, N_{j+1})$ in the positive samples and expressed as: 
$$
\tag{1}
P^+[i, N_j, N_{j+1}] = \displaystyle\frac{\text{freq}(i, N_j, N_{j+1})}{\sum\limits_{N_{j+1}}\text{freq}(i-1, N_j, N_{j+1})}, \ \ i = 2, \cdots, \lambda, \ \ j = 1, 2, 3, 4
$$

The conditional distribution probabilities of negative samples $P^-[i, N_j, N_{j+1}]$ are defined likewise: 
$$
\tag{2}
P^-[i, N_j, N_{j+1}] = \displaystyle\frac{\text{freq}(i, N_j, N_{j+1})}{\sum\limits_{N_{j+1}}\text{freq}(i-1, N_j, N_{j+1})},  \ \ i = 2, \cdots, \lambda, \ \ j = 1, 2, 3, 4
$$
Then we put the base distribution probabilities of the first position $P^+[1, N]$ into 1D-arrays $P_0^+$ and $P_0^-$ : 
$$
\tag{3}
P_0^+[1, N_j] = \displaystyle\frac{\text{freq}(1, N_j)}{\sum\limits_{N_{j}}\text{freq}(1, N_j)}
$$

$$
\tag{4}
P_0^-[1, N_j] = \displaystyle\frac{\text{freq}(1, N_j)}{\sum\limits_{N_{j}}\text{freq}(1, N_j)}
$$

The base distribution of positive samples nicely dovetails with the currently accepted splice site pattern, as shown in Fig. 3, which further proves the validity of our proposed method: the conditional probability matrices are capable of reflecting special functional signals in nucleotide sequences. 

<center class = "half" >     <img src = "D:\Project\SplicePredictor\Pics\donor\ssheatmap.png" width = "500">    <img src = "D:\Project\SplicePredictor\Pics\donor\ssheatmap_neg.png" width = "500"></center>

<center class = "half" >     <img src = "D:\Project\SplicePredictor\Pics\donor\asheatmap.png" width = "500">    <img src = "D:\Project\SplicePredictor\Pics\donor\asheatmap_neg.png" width = "500"></center>

<center class = "half" >     <img src = "D:\Project\SplicePredictor\Pics\acceptor\ssheatmap.png" width = "500">    <img src = "D:\Project\SplicePredictor\Pics\acceptor\asheatmap.png" width = "500"></center>

**Figure 3** Base distribution represented as heatmaps for donor splice sites. (a)(b) Single base distribution of positive donor \& negative samples. Aside from two conservative sites GT (labeled with "site"), Positions (-2, -1, +1, +2, +3) shows additional conservation of adjacent sites of a splice site. For example, it is attractive that the odds reach 84\% of position +3 being a guanine. (c)(d) Adjacent base distribution of positive donor \& negative samples which shows the correspondence between bases more clearly compared with the single base heatmaps. (e)(f) Base distribution of positive acceptor samples. The polypyrimidine tract can be observed (-5, -4, -3) likewise, with the sum of $P(C)$ and $P(T)$ exceeds 80\%, which is part of the powerful acceptor signal. 

### 2.4 Prediction

We apply the P matrices above to make judgment of splice sites in unknown sequences. To be specific, we set a sliding window to extract sequences of every available position of the testing set, and score them with a scoring function $S(X)$. For a binary classifier, the scoring formula is a log-odds ratio as: 
$$
\tag{5}
S(X) = \ln \displaystyle\frac{P^+(X)}{P^-(X)} 
= \ln \displaystyle\frac{P_0^+[1, N_1]}{P_0^-[1, N_1]} + \sum\limits_{i=2}^\lambda\ln \displaystyle\frac{P^+[i, N_{i-1}, N_i]}{P^-[i, N_{i-1}, N_i]}
$$
Since there is zeros in the probability matrices, we set  $P = 10^{-6}$  to avoid division-by-zero error which, in the meantime, guarantees a higher penalty for a sequence including a $P^+(X) = 0$ site, and vice versa. By this way, we score all sequence txts of the testing set and get the score distributions of donor and acceptor predicting models, as is shown in Fig. 4. The distributions are conducive to the selection of thresholds at the following steps. 

![density](D:\Project\SplicePredictor\Pics\density.svg)

**Figure 4** Density of sequence scores predicted separately by donor (blue) and acceptor (coral) WAMs. Compared to the histograms Rodger \cite{staden1984computer} provided earlier, there are a lot of low scores in our testing consequence. This is caused by the sliding selection of testing sequences without filtering pseudo ones with the conservative sites on purpose. It can also be observed that low and high scores are widely separated, hence can be easily distinguished. Comparing donor and acceptor scores latitudinally, we can see difference between the sequence pattern of donor and acceptor sites, but it is small enough to allow the acceptor prediction without changing the framework of our approach. 

Transformation from scores to predicting results needs the comparison. We can filter the true positive sites we need from batches of scores by taking different thresholds. It is necessary to exercise caution in selecting the threshold $C$. A large $C$ will exclude potential positive sites, while a small $C$ misclassifies negative sites as positive. Hence an appropriate threshold is a tradeoff based on the specificity and sensitivity of a model. For the threshold optima selection, see 4 Results for details. 

## 3 Experiments

### 3.1 Data

We conduct our experiment on the eukaryotic gene sequence dataset Kulp \& Reese \cite{reese1997improved} and Burset \& Guigo \cite{burset1996evaluation}. Human genome sequence dataset Kulp \& Reese (KR set) is used as training set which contains 462 sequence text files, each records the name, length, CDS terminal points and the segment. 2,381 donor sites and 2,381 acceptor sites are extracted from the KR set. Vertebrate genome dataset Burset \& Guigo (BG set) is used as testing set which contains 570 sequence text files with a similar format, except for a lack of the sequence length. 

The KR and BG set is open access and you can get the entire dataset at https://www.fruitfly.org/sequence/human-datasets.html and http://www1.imim.es/databases/genomics96/. 

### 3.2 Metrics

Our model accuracy measures are given by (6) -- (12): 
$$
\tag{6}
\text{Precision} = \displaystyle\frac{\text{TP}}{\text{TP} + \text{FP}}
$$

$$
\tag{7}
\text{Recall} = \displaystyle\frac{\text{TP}}{\text{TP} + \text{FN}}
$$

$$
\tag{8}
\text{FPR} = \displaystyle\frac{\text{FP}}{\text{TN} + \text{FP}}
$$

$$
\tag{9}
\text{TPR} = \displaystyle\frac{\text{TP}}{\text{TP} + \text{FN}}
$$

$$
\tag{10}
\text{F1-Score} = \displaystyle\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

where $\text{TP}$, $\text{FP}$, $\text{TN}$, $\text{FN}$ are metrics of the confusion matrix \cite{stehman1997selecting}. Precision-Recall curves and ROC curves \cite{powers2020evaluation}\cite{fawcett2006introduction} are plotted to make the performance of our model more intuitive. We also calculate areas under the curves by: 
$$
\tag{11}
\text{AP} = \int_0^1 P(R)\text{d}R = \sum\limits_{i=1}^n P_i\Delta R_i
$$

$$
\tag{12}
\text{AUC} = \int_0^1 T(F)\text{d}F = \sum\limits_{i=1}^n T_i\Delta F_i
$$

where $\text{AP}$ summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight \cite{zhu2004recall}. $\text{AUC}$ is equal to the probability that the model will rank a randomly chosen positive sample higher than a randomly chosen negative one (assuming that "positive" ranks higher than "negative"), where $F$ denotes false positive rate and $T$ denotes true positive rate \cite{fawcett2006introduction}. 

### 3.3 Implementation

We encapsulate the WAM model in the class `Wam` which is derived from the base class `Base` in `./Model/basemodel.py`. Additionally, intermediate class `Ssm` is created for the shared parts of WMM and WAM. Sequences are extracted by `./Utils/extract.py` and saved temporarily in an `Sequence` object. All of the statistical graphs involved in this paper are drawn by the scripts in `./Utils` using Matplotlib \cite{Hunter:2007}, Seaborn \cite{Waskom2021} \& Weblogo, and saved in `./Pics`. We evaluate the model using Scikit-Learn for confusion matrices, precision-recall pairs and FPR-TPR pairs \cite{pedregosa2011scikit}. These tools saved considerable time for model training and prediction. Models can be easily saved or loaded by methods `save_model()` and `load_model()`. All the components have their corresponding interface methods provided in the aforementioned classes. For the code implementation details of WAM (also WMM), see `./Model/wam.py `. 

Training \& Predicting process is operated by Ubuntu 18.04.5 LTS on 16 CPU cores with 2 threads each. The source code is available on GitHub and can be obtained from https://github.com/Newiz430/SplicePredictor. 

## 4 Results

### 4.1 Donor Site Prediction

**Comparison Studies** We compared the performance of our WAM model with the conventional WMM model by Rodger, et al. \cite{staden1984computer} We evaluated two models under a set of different thresholds and the results are shown in Fig. 5. These results validate that WAM outperforms WMM in pattern identification, as  We also found the thresholds with the highest F1 score which indicates the best predicting result, as is shown in Fig. 6. 

<center class = "half" >
    <img src = "D:\Project\SplicePredictor\Pics\donor\WMM_WAM_prcurve.png" width = "500">
    <img src = "D:\Project\SplicePredictor\Pics\donor\WMM_WAM_roccurve.png" width = "500">
</center>
**Figure 5** Measuring WMM \& WAM models on part of the BG set. (a) Precision-Recall curves plotted by using a bunch of thresholds. Average Precision is marked at the figure legend, which represents areas under the P-R curves. Both two models performed well on the given training data with impressive predicting precisions, aside from which WAM model has a much higher AP (0.9759) than WMM (0.9641), which means that WAM predicts fewer false positive results under the circumstance of the correct prediction of the same amount of positive sites. (b) ROC curves with AUC marked at the figure legend, which represents areas under the ROC. The results conforms to P-R curves' as expected, although it's harder to tell the difference between two models. 

We used the training set of the same size and predicting procedure for both two models, and performed them on the same set of testing data with the best threshold Fig. 6 pointing out. In Table 1,  we present the stats of accuracy for both models. It indicates that WAM improves the overall signal predicting effect within an approximate predicting time. WAM shows a +2.27\% improvement on F1-score from 0.9061 up to 0.9267. 

<center class = "half" >
    <img src = "D:\Project\SplicePredictor\Pics\donor\WMM_threshold.png" width = "500">
    <img src = "D:\Project\SplicePredictor\Pics\donor\WAM_threshold.png" width = "500">
</center>
**Figure 6** Searching for the best thresholds for donor prediction. F1-score (10) is a balanced metric between  precision and recall which expresses best performance of models. We have found the maximum F1 points and values for both models: (a) WMM, (2.26, 0.91) ; (b) WAM, (3.09, 0.93). 

**Table 1** Performance of the proposed WAM model against WMM on donor sites. The argmax thresholds are assigned to these models to get the best metrics. Run time represents the seconds cost in the predicting step. 

| Method                 | Precision  | Recall (TPR) | FPR        | F1-score   | Run time (s) |
| ---------------------- | ---------- | ------------ | ---------- | ---------- | ------------ |
| WMM (threshold = 2.26) | 0.8503     | **0.9697**   | 0.0085     | 0.9061     | **1.6614**   |
| WAM (threshold = 3.09) | **0.9006** | 0.9543       | **0.0053** | **0.9267** | 1.7940       |

### 4.2 Acceptor Site Prediction

We did the same experiment on acceptor sites. Consequences are displayed in Fig. 7 - 8 and Table 2. 

<center class = "half" >
    <img src = "D:\Project\SplicePredictor\Pics\acceptor\WMM_WAM_prcurve.svg" width = "500">
    <img src = "D:\Project\SplicePredictor\Pics\acceptor\WMM_WAM_roccurve.svg" width = "500">
</center>
**Figure 7** Precision-Recall curves (a) and ROC curves (b) for acceptor signal. WAM Model used for acceptor seems to have a lower predicting ability relative to the one for donor. This can be explained by our hypothesis that adjacent dependencies for acceptor influences the accuracy slighter than donor's. 

<center class = "half" >
    <img src = "D:\Project\SplicePredictor\Pics\acceptor\WMM_threshold.svg" width = "500">
    <img src = "D:\Project\SplicePredictor\Pics\acceptor\WAM_threshold.svg" width = "500">
</center>
**Figure 8** Searching for the best thresholds for acceptor prediction. We have found the maximum F1 points and values for both models: (a) WMM, (2.69, 0.81) ; (b) WAM, (2.54, 0.82). 

**Table 2** Performance of the proposed WAM model against WMM on acceptor sites. The argmax thresholds are assigned to these models to get the best metrics. Run time represents the seconds cost in the predicting step. 

| Method                 | Precision  | Recall (TPR) | FPR        | F1-score   | Run time (s) |
| ---------------------- | ---------- | ------------ | ---------- | ---------- | ------------ |
| WMM (threshold = 2.69) | 0.7385     | **0.9038**   | 0.0160     | 0.8128     | **1.3336**   |
| WAM (threshold = 2.54) | **0.7554** | **0.9038**   | **0.0146** | **0.8229** | 1.4286       |

For acceptor, WAM shows an +1.24\% improvement on F1-score from 0.8128 up to 0.8229. In a nutshell, data certifies that our model is available for high precision predictions, which neither costs much time nor quantities of computing resources. 

## 5 Discussion

Overall, we formulate and re-implement an application of WAM model by training it on the Kulp \& Reese dataset. We compare its performance against the conventional WMM, and successfully prove its superiority on the accuracy of predicting donor \& acceptor splice sites. 

As a matter of fact, there are still some blemishes in our methods which need to be taken serious consideration. We only sampled a fraction of data for matrix construction thus our model may not attain its best performance with the given training set. We ignored the odds of base indels in signal sequences. We omitted unambiguous bases at the beginning of our work which is likely to be part of the splice site patterns. What's more, we only tried one single feature selection tactic limited by the deadline (our model is actually designed for feature sequences of different lengths as input). 

In the future, we will explore deeper into the frequency-based nucleotide pattern finding methods with better generality, efficiency and practicality, aside from addressing the issues above. 

## Acknowledgement

This work was supported by Prof. Zhou from College of Life Science \& Technology, Huazhong University of Science and Technology, and Wuhan National Laboratory for Optoelectronics for providing computing resources. Also acknowledge our classmates for helpful suggestions \& corrections. 

## References

```latex
@article{burge1997prediction,
  title={Prediction of complete gene structures in human genomic DNA},
  author={Burge, Chris and Karlin, Samuel},
  journal={Journal of molecular biology},
  volume={268},
  number={1},
  pages={78--94},
  year={1997},
  publisher={Elsevier}
}

@article{clancy2008rna,
  title={RNA splicing: introns, exons and spliceosome},
  author={Clancy, Suzanne and others},
  journal={Nature Education},
  volume={1},
  number={1},
  pages={31},
  year={2008}
}

@article{black2003mechanisms,
  title={Mechanisms of alternative pre-messenger RNA splicing},
  author={Black, Douglas L},
  journal={Annual review of biochemistry},
  volume={72},
  number={1},
  pages={291--336},
  year={2003},
  publisher={Annual Reviews 4139 El Camino Way, PO Box 10139, Palo Alto, CA 94303-0139, USA}
}

@book{lodish2008molecular,
  title={Molecular cell biology},
  author={Lodish, Harvey and Berk, Arnold and Kaiser, Chris A and Kaiser, Chris and Krieger, Monty and Scott, Matthew P and Bretscher, Anthony and Ploegh, Hidde and Matsudaira, Paul and others},
  year={2008},
  publisher={Macmillan}
}

@article{schneider1990sequence,
  title={Sequence logos: a new way to display consensus sequences},
  author={Schneider, Thomas D and Stephens, R Michael},
  journal={Nucleic acids research},
  volume={18},
  number={20},
  pages={6097--6100},
  year={1990},
  publisher={Oxford University Press}
}

@article{shannon1948mathematical,
  title={A mathematical theory of communication},
  author={Shannon, Claude E},
  journal={The Bell system technical journal},
  volume={27},
  number={3},
  pages={379--423},
  year={1948},
  publisher={Nokia Bell Labs}
}

@article{schneider1986information,
  title={Information content of binding sites on nucleotide sequences},
  author={Schneider, Thomas D and Stormo, Gary D and Gold, Larry and Ehrenfeucht, Andrzej},
  journal={Journal of molecular biology},
  volume={188},
  number={3},
  pages={415--431},
  year={1986},
  publisher={Elsevier}
}

@article{sakharkar2004distributions,
  title={Distributions of exons and introns in the human genome},
  author={Sakharkar, Meena Kishore and Chow, Vincent TK and Kangueane, Pandjassarame},
  journal={In silico biology},
  volume={4},
  number={4},
  pages={387--393},
  year={2004},
  publisher={IOS press}
}

@article{black2003mechanisms,
  title={Mechanisms of alternative pre-messenger RNA splicing},
  author={Black, Douglas L},
  journal={Annual review of biochemistry},
  volume={72},
  number={1},
  pages={291--336},
  year={2003},
  publisher={Annual Reviews 4139 El Camino Way, PO Box 10139, Palo Alto, CA 94303-0139, USA}
}
@article{staden1984computer,
  title={Computer methods to locate signals in nucleic acid sequences},
  author={Staden, Rodger},
  year={1984},
  publisher={Oxford University Press}
}

@article{chen2005prediction,
  title={Prediction of splice sites with dependency graphs and their expanded bayesian networks},
  author={Chen, Te-Ming and Lu, Chung-Chin and Li, Wen-Hsiung},
  journal={Bioinformatics},
  volume={21},
  number={4},
  pages={471--482},
  year={2005},
  publisher={Oxford University Press}
}
@article{duan2008position,
  title={Position-specific residue preference features around the ends of helices and strands and a novel strategy for the prediction of secondary structures},
  author={Duan, Mojie and Huang, Min and Ma, Chuang and Li, Lun and Zhou, Yanhong},
  journal={Protein science},
  volume={17},
  number={9},
  pages={1505--1512},
  year={2008},
  publisher={Wiley Online Library}
}
@inproceedings{ryen2008splice,
  title={Splice site prediction using artificial neural networks},
  author={Ryen, Tom and Eftes, Trygve and Kjosmoen, Thomas and Ruoff, Peter and others},
  booktitle={International Meeting on Computational Intelligence Methods for Bioinformatics and Biostatistics},
  pages={102--113},
  year={2008},
  organization={Springer}
}
@article{zhang1993weight,
  title={A weight array method for splicing signal analysis},
  author={Zhang, MO and Marr, TG},
  journal={Bioinformatics},
  volume={9},
  number={5},
  pages={499--509},
  year={1993},
  publisher={Oxford University Press}
}
@article{dash2001modeling,
  title={Modeling DNA splice regions by learning Bayesian networks},
  author={Dash, Denver and Gopalakrishnan, Vanathi},
  year={2001},
  publisher={Citeseer}
}
@article{reese1997improved,
  title={Improved splice site detection in Genie},
  author={Reese, Martin G and Eeckman, Frank H and Kulp, David and Haussler, David},
  journal={Journal of computational biology},
  volume={4},
  number={3},
  pages={311--323},
  year={1997}
}
@article{burset1996evaluation,
  title={Evaluation of gene structure prediction programs},
  author={Burset, Moises and Guigo, Roderic},
  journal={genomics},
  volume={34},
  number={3},
  pages={353--367},
  year={1996},
  publisher={Elsevier}
}
@article{stehman1997selecting,
  title={Selecting and interpreting measures of thematic classification accuracy},
  author={Stehman, Stephen V},
  journal={Remote sensing of Environment},
  volume={62},
  number={1},
  pages={77--89},
  year={1997},
  publisher={Elsevier}
}
@article{powers2020evaluation,
  title={Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation},
  author={Powers, David MW},
  journal={arXiv preprint arXiv:2010.16061},
  year={2020}
}
@article{fawcett2006introduction,
  title={An introduction to ROC analysis},
  author={Fawcett, Tom},
  journal={Pattern recognition letters},
  volume={27},
  number={8},
  pages={861--874},
  year={2006},
  publisher={Elsevier}
}
@article{zhu2004recall,
  title={Recall, precision and average precision},
  author={Zhu, Mu},
  journal={Department of Statistics and Actuarial Science, University of Waterloo, Waterloo},
  volume={2},
  number={30},
  pages={6},
  year={2004}
}
@article{Hunter:2007,
  Author    = {Hunter, J. D.},
  Title     = {Matplotlib: A 2D graphics environment},
  Journal   = {Computing in Science \& Engineering},
  Volume    = {9},
  Number    = {3},
  Pages     = {90--95},
  abstract  = {Matplotlib is a 2D graphics package used for Python for
  application development, interactive scripting, and publication-quality
  image generation across user interfaces and operating systems.},
  publisher = {IEEE COMPUTER SOC},
  doi       = {10.1109/MCSE.2007.55},
  year      = 2007
}
@article{Waskom2021,
  doi = {10.21105/joss.03021},
  url = {https://doi.org/10.21105/joss.03021},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {60},
  pages = {3021},
  author = {Michael L. Waskom},
  title = {seaborn: statistical data visualization},
  journal = {Journal of Open Source Software}
}
@article{pedregosa2011scikit,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={the Journal of machine Learning research},
  volume={12},
  pages={2825--2830},
  year={2011},
  publisher={JMLR. org}
}
```



