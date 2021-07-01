#!/usr/bin/env python

from typing import List, Tuple, Union
import abc
import os
import re
from random import seed, sample
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from Utils.extract import Sequence

seed(1) # This guarantees the same testing set among all model predictions


class Base(metaclass=abc.ABCMeta):

    def __init__(self,
                 data: Sequence,
                 site_type: str) -> None:
        """Base class of all 4 splice predictors. This cannot be instantiated as Base is virtual.

        Parameters
        ----------
        data: A Utils.extract.Sequence instance
            Contains the signal sequences we need to train our model.
            Donor & acceptor sites are saved in data.donor_site and data.acceptor_site,
            whose negative control sequences are in data.neg_donor_site and neg_acceptor_site.
            (The two neg site lists are the same by default.)

        site_type: str (one of "donor", "acceptor")
            Specifies the training sequences we will use to train the model.

        References
        ----------
        [1] Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830, 2011
        """

        if site_type not in {"donor", "acceptor"}:
            raise Exception("ERROR: Wrong site type.")

        self.name = None
        self.data = data
        self.feat_length = data.up_sites + data.down_sites + 2
        self.site_type = site_type
        self.encoding_map = None

    @staticmethod
    def _remove_ambiguous_bases(seq: str) -> str:

        return re.sub("[^ACGTacgt]", '', seq)

    @abc.abstractmethod
    def predict(self,
                seq: Union[str, np.ndarray],
                threshold: Union[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Find potential splice sites in a single sequence txt, and return the positions. """

        pass

    def calc_scores(self,
                    testset: Sequence,
                    threshold: Union[int, float] = 0,
                    testset_ratio: float = 1,
                    batch_of_neg_samples: int = 20,
                    load_scores: bool = False,
                    save_scores: bool = True,
                    verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate scores for the entire testing data. """

        if testset_ratio <= 0 or testset_ratio > 1:
            raise Exception("ERROR: invalid ratio of testing set.")

        if load_scores:
            # Loading scores (unused)
            scores = np.load("../Model/tmp/{}_{}_scores.npy".format(self.name, self.site_type))
            preds = np.load("../Model/tmp/{}_{}_preds.npy".format(self.name, self.site_type))
            reals = np.load("../Model/tmp/{}_{}_reals.npy".format(self.name, self.site_type))

        else:
            scores = []
            preds = []
            reals = []

            bar = tqdm(os.listdir(testset.filepath)[:int(testset_ratio * len(os.listdir(testset.filepath)))])
            for txt in bar:
                bar.set_postfix(step="Calculating scores")
                donor_site, neg_donor_site, acceptor_site, neg_acceptor_site = testset.extract_single_txt(txt)
                # We only use a fraction of negative samples
                neg_donor_site = sample(neg_donor_site, batch_of_neg_samples * len(donor_site)) \
                    if len(neg_donor_site) > batch_of_neg_samples * len(donor_site) else neg_donor_site
                neg_acceptor_site = sample(neg_acceptor_site, batch_of_neg_samples * len(acceptor_site)) \
                    if len(neg_acceptor_site) > batch_of_neg_samples * len(acceptor_site) else neg_acceptor_site

                test_seq = np.array(donor_site + neg_donor_site) if self.site_type == "donor" \
                    else np.array(acceptor_site + neg_acceptor_site)
                res = self.predict(test_seq, threshold)

                preds.extend(res[1])
                scores.extend(res[2])
                # Generate ground truth
                real = np.repeat(False, len(test_seq))
                real[: len(donor_site if self.site_type == "donor" else acceptor_site)] = True
                real = np.delete(real, res[3])  # Abandon invalid sequences
                reals.extend(real)

                if verbose:
                    print("splice site:", donor_site)
                    print("score:", res[2])
                    print("pred:", res[1])
                    print("real:", real)

            # Ignoring NaN
            nan = np.where(np.isnan(np.array(scores)))[0]
            scores = np.delete(np.array(scores), nan)
            preds = np.delete(np.array(preds), nan)
            reals = np.delete(np.array(reals), nan)

            # Save scores (unused)
            if save_scores:
                np.save("../Model/tmp/{}_{}_scores.npy".format(self.name, self.site_type), np.array(scores))
                np.save("../Model/tmp/{}_{}_preds.npy".format(self.name, self.site_type), np.array(preds))
                np.save("../Model/tmp/{}_{}_reals.npy".format(self.name, self.site_type), np.array(reals))

        return scores, preds, reals

    @staticmethod
    def get_plot_data(scores: np.ndarray,
                      reals: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
        """Return data for plotting. No need for a threshold. """

        p_list, r_list, thr_list = metrics.precision_recall_curve(reals, scores, pos_label=True)
        fpr_list, tpr_list, _ = metrics.roc_curve(reals, scores, pos_label=True, drop_intermediate=False)
        ap = metrics.average_precision_score(reals, scores, pos_label=True)
        auc = metrics.roc_auc_score(reals, scores)

        return thr_list, p_list, r_list, ap, fpr_list, tpr_list, auc

    @staticmethod
    def evaluate(preds: np.ndarray,
                 reals: np.ndarray) -> Tuple[np.ndarray, float, float, float, float, float]:
        """Evaluate the model by predicting the testing set. """

        cm = metrics.confusion_matrix(reals, preds, labels=[False, True])
        r = metrics.recall_score(reals, preds)
        p = metrics.precision_score(reals, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (tn + fp)
        tpr = tp / (tp + fn)
        f1 = metrics.f1_score(reals, preds)

        return cm, r, p, fpr, tpr, f1

    @abc.abstractmethod
    def save_model(self,
                   save_path: str = "../Model",
                   model_name: str = "Model") -> None:
        """Save model as .model file. """

        pass

    @abc.abstractmethod
    def load_model(self,
                   load_path: str = "../Model",
                   model_name: str = "Model") -> None:
        """Load model from a .model file. """

        pass