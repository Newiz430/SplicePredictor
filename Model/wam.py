#!/usr/bin/env python

from typing import Any, List, Tuple, Union
import abc
import os
from random import sample
import numpy as np
import joblib

from Model.basemodel import Base
from Utils.extract import Sequence

__all__ = [
    "Wmm",
    "Wam",
]


class Ssm(Base):

    def __init__(self,
                 data: Sequence,
                 site_type: str) -> None:
        """Base class of WMM and WAM. """

        super(Ssm, self).__init__(data, site_type)

        if site_type not in {"donor", "acceptor"}:
            raise Exception("ERROR: Wrong site type.")

        self.encoding_map = str.maketrans({'A': '0', 'C': '1', 'G': '2', 'T': '3',
                                           'a': '0', 'c': '1', 'g': '2', 't': '3'})

        self.p_plus = None
        self.p_minus = None
        self.p0_plus = None
        self.p0_minus = None

    @abc.abstractmethod
    def _calc_p(self,
                type: str,
                encoding: List[List[int]]) -> Any:
        """Calculate probability matrices. """

        pass

    @abc.abstractmethod
    def _score(self,
               s: str) -> float:
        """Score a sequence. This function varies for different methods. """

        pass

    def encode(self,
               s: str) -> List[int]:
        """Encode DNA sequences into a figure list. """

        return list(map(int, self._remove_ambiguous_bases(s).translate(self.encoding_map)))

    @abc.abstractmethod
    def fit(self) -> None:
        """Process raw sequences & Generate p+ & p- matrices. """

        pass

    def predict(self,
                seq: Union[str, np.ndarray],
                threshold: Union[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Find potential splice sites in a single sequence txt, and return the positions. """

        if isinstance(seq, str):
            # Sliding & Scoring
            length = len(seq)
            score = np.array([self._score(seq[i: i + self.feat_length]) for i in range(0, length - self.feat_length)])
        else:
            score = np.array([self._score(s) for s in seq])
        seq_pred = score > threshold

        return np.nonzero(seq_pred == True)[0] + self.data.up_sites, seq_pred, score, []

    def save_model(self,
                   save_path: str = "../Model",
                   model_name: str = "WAM") -> None:

        if not save_path:
            raise Exception("ERROR: Invalid save path.")

        value = (self.p_plus, self.p_minus) if self.name == "WMM" else (self.p0_plus, self.p0_minus,
                                                                        self.p_plus, self.p_minus)
        joblib.dump(filename=os.path.join(save_path, "{}/{}.model".format(self.site_type, model_name)), value=value)
        print("Done! Model saved in {}/{}.model.".format(save_path, "{}/{}".format(self.site_type, model_name)))

    def load_model(self,
                   load_path: str = "../Model",
                   model_name: str = "WAM") -> None:

        if not os.path.join(load_path, "{}/{}.model".format(self.site_type, model_name)):
            raise Exception("ERROR: Invalid file name.")

        model = joblib.load(os.path.join(load_path, "{}/{}.model".format(self.site_type, model_name)))

        if self.name == "WMM":
            self.p_plus, self.p_minus = model
        else:
            self.p0_plus, self.p0_minus, self.p_plus, self.p_minus = model
        print("Done! Model loaded from {}/{}.model.".format(load_path, "{}/{}".format(self.site_type, model_name)))


class Wmm(Ssm):

    def __init__(self,
                 data: Sequence,
                 site_type: str) -> None:
        """WMM model for splice prediction. """

        super(Wmm, self).__init__(data, site_type)

        self.name = "WMM"
        self.p_plus = np.zeros((self.feat_length, 4))
        self.p_minus = np.zeros((self.feat_length, 4))

    def _calc_p(self, type: str, encoding: List[List[int]]) -> np.ndarray:

        if type not in ("plus", "minus"):
            raise Exception("ERROR: Invalid probability matrix type.")
        p = self.p_plus if type == "plus" else self.p_minus

        for eseq in encoding:
            if len(eseq) < self.feat_length:
                continue  # Omit unambiguous bases
            for i, s in enumerate(eseq):
                p[i][s] += 1
        return np.array([p[i] / np.array([sum(p[i])] * 4) for i in range(self.feat_length)])

    def _score(self, s: str) -> float:

        es = self.encode(s)
        if len(es) < self.feat_length:
            score = np.nan
        else:
            p = []
            for i in range(self.feat_length):
                ps = self.p_plus[i][es[i]] / max(self.p_minus[i][es[i]], 1e-6)
                p.append(np.log(max(ps, 1e-6)))

            score = sum(p)
        return score

    def fit(self) -> None:

        plus_sites = self.data.donor_site * 2 if self.site_type == "donor" else self.data.acceptor_site * 2

        # Encode bases
        plus_encoding = [self.encode(seq) for seq in plus_sites]
        # Get the probability matrix
        self.p_plus = self._calc_p("plus", plus_encoding)

        # If there is not p_minus, create one
        if not np.any(self.p_minus):
            minus_sites = self.data.neg_donor_site if self.site_type == "donor" else self.data.neg_acceptor_site
            # Only part of the negative sites is used
            minus_encoding = [self.encode(seq) for seq in sample(minus_sites, 5000)]

            self.p_minus = self._calc_p("minus", minus_encoding)


class Wam(Ssm):

    def __init__(self,
                 data: Sequence,
                 site_type: str) -> None:
        """WAM model for splice prediction. """

        super(Wam, self).__init__(data, site_type)

        self.name = "WAM"
        # p of initial bases
        self.p0_plus = np.zeros(4)
        self.p0_minus = np.zeros(4)
        # p of adjacent bases
        self.p_plus = np.zeros((self.feat_length - 1, 4, 4))
        self.p_minus = np.zeros((self.feat_length - 1, 4, 4))

    def _calc_p(self,
                type: str,
                encoding: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:

        if type not in ("plus", "minus"):
            raise Exception("ERROR: Invalid probability matrix type.")
        (p0, p) = (self.p0_plus, self.p_plus) if type == "plus" else (self.p0_minus, self.p_minus)

        for eseq in encoding:
            if len(eseq) < self.feat_length:
                continue # Omit unambiguous bases
            p0[eseq[0]] += 1
            for i in range(1, self.feat_length):
                p[i - 1][eseq[i - 1]][eseq[i]] += 1
        p0 /= np.array([sum(p0)] * 4)
        p = np.array([[np.zeros(4) if sum(p[i][j]) == 0 else p[i][j] / ([sum(p[i][j])] * 4) for j in range(4)]
                      for i in range(self.feat_length - 1)])
        return p, p0

    def _score(self,
               s: str) -> float:

        es = self.encode(s)
        if len(es) < self.feat_length:
            score = np.nan
        else:
            # score = p_front - p_background
            p0 = self.p0_plus[es[0]] / max(self.p0_minus[es[0]], 1e-6)
            p0 = np.log(1e-6 if p0 < 1e-6 else p0) # Dealing with zeroes
            p = []
            for i in range(1, self.feat_length):
                ps = self.p_plus[i - 1][es[i - 1]][es[i]] / max(self.p_minus[i - 1][es[i - 1]][es[i]], 1e-6)
                p.append(np.log(max(ps, 1e-6)))

            score = p0 + sum(p)
        return score

    def fit(self) -> None:

        plus_sites = self.data.donor_site * 2 if self.site_type == "donor" else self.data.acceptor_site * 2

        # Encode bases
        plus_encoding = [self.encode(seq) for seq in plus_sites]
        # Get the probability matrix
        self.p_plus, self.p0_plus = self._calc_p("plus", plus_encoding)

        # If there is not p_minus, create one
        if not np.any(self.p0_minus):
            minus_sites = self.data.neg_donor_site if self.site_type == "donor" else self.data.neg_acceptor_site
            # Only part of the negative sites is used
            minus_encoding = [self.encode(seq) for seq in sample(minus_sites, 5000)]

            self.p_minus, self.p0_minus = self._calc_p("minus", minus_encoding)


if __name__ == "__main__":

    import time

    TRAINING_PATH = "../Data/Training Set"
    TESTING_PATH = "../Data/Testing Set"

    # Set params
    site_type = "donor"
    wmm_threshold = 2.69 # Best threshold
    wam_threshold = 2.54 # Best threshold
    model_path = '' # Use '.' to activate load-model mode

    seq = Sequence(filepath=TRAINING_PATH, type="train", up_sites=5, down_sites=5)
    test_seq = Sequence(filepath=TESTING_PATH, type="test", up_sites=5, down_sites=5)

    if model_path:
        wmm = Wmm(seq, site_type)
        wam = Wam(seq, site_type)
        wmm.load_model(model_path, model_name="WMM")
        wam.load_model(model_path, model_name="WAM")

    else:
        wmm = Wmm(seq, site_type)
        wmm.fit()
        wam = Wam(seq, site_type)
        wam.fit()

    wmm.save_model(model_name="WMM")
    wam.save_model(model_name="WAM")

    print("WMM Predicting...")
    start = time.time()
    _, preds, reals = wmm.calc_scores(test_seq, wmm_threshold, testset_ratio=1)
    cm, r, p, fpr, tpr, f1 = wmm.evaluate(preds, reals)
    tn, fn, fp, tp = cm.ravel()
    print("WMM: threshold={}, tn={}, fp={}, fn={}, tp={}, \nrecall={}, precision={}, fpr={}, tpr={}, f1={}".format(
        wmm_threshold, tn, fp, fn, tp, r, p, fpr, tpr, f1))
    end = time.time()
    print("WMM runtime: {}".format(end - start))

    print("WAM Predicting...")
    start = time.time()
    _, preds, reals = wam.calc_scores(test_seq, wam_threshold, testset_ratio=1)
    cm, r, p, fpr, tpr, f1 = wam.evaluate(preds, reals)
    tn, fp, fn, tp = cm.ravel()
    print("WAM: threshold={}, tn={}, fp={}, fn={}, tp={}, \nrecall={}, precision={}, fpr={}, tpr={}, f1={}".format(
        wam_threshold, tn, fp, fn, tp, r, p, fpr, tpr, f1))
    end = time.time()
    print("WAM runtime: {}".format(end - start))
