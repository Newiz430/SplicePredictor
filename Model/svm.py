#!/usr/bin/env python

from typing import Tuple, List, Union
import os
from random import sample
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# from thundersvm import SVC # Use this with CUDA settings for SVM with a faster speed (highly recommended)

from Model.basemodel import Base
from Utils.extract import Sequence

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Svm(Base):

    def __init__(self,
                 data: Sequence,
                 site_type: str) -> None:
        """A support vector machine for splice prediction. """

        super(Svm, self).__init__(data, site_type)

        self.name = "SVM"
        self.encoding_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
        self.encoding_seq = None
        self.neg_encoding_seq = None

        self.svm = None

    def _encode_a_single_seq(self,
                             seq: str) -> np.ndarray:

        enc = []
        seq = self._remove_ambiguous_bases(seq)
        for b in seq:
            enc.extend(np.identity(4)[self.encoding_map.get(b)])
        return np.asarray(enc)

    def encode(self) -> None:
        """Encode sequences by bases. """

        (pos_sites, neg_sites) = (self.data.donor_site, self.data.neg_donor_site) \
            if self.site_type == "donor" else (self.data.acceptor_site, self.data.neg_acceptor_site)

        # Encoding (Remove sequences with ambiguous bases)
        self.encoding_seq = np.array([self._encode_a_single_seq(site) for site in pos_sites
                                      if len(self._remove_ambiguous_bases(site)) == self.feat_length] * 2)
        self.neg_encoding_seq = np.array(sample([self._encode_a_single_seq(site) for site in neg_sites
            if len(self._remove_ambiguous_bases(site)) == self.feat_length], 5000))

        print("Sequences encoded successfully with {} * positive sequences and {} * negative sequences."
              .format(len(self.encoding_seq), len(self.neg_encoding_seq)))

    def set_params(self,
                   C: float = 1.,
                   kernel: str = 'rbf',
                   degree: int = 3,
                   gamma: str = "scale",
                   coef0: float = 0.,
                   probability=True,
                   cache_size: int = 200,
                   class_weight: dict = None,
                   verbose: bool = True,
                   random_state: int = 1) -> None:
        """Set parameters for the SVM model. """

        if kernel not in {"linear", "poly", "rbf"}:
            raise Exception("ERROR: Invalid kernel name.")

        self.svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, probability=probability,
                  cache_size=cache_size, class_weight=class_weight if class_weight else "balanced",
                  verbose=verbose, random_state=random_state)

    def get_params(self) -> dict:
        """Get parameters of the SVM model. """

        return self.svm.get_params()

    def _integrate_train_data(self) -> Tuple[np.ndarray, np.ndarray]:

        X = np.vstack((self.encoding_seq, self.neg_encoding_seq))
        y = np.repeat((0, 1), (len(self.encoding_seq), len(self.neg_encoding_seq)))

        return X, y

    @staticmethod
    def _standardize(X: np.ndarray) -> np.ndarray:

        return StandardScaler().fit_transform(X)

    def fit(self,
            standardize: bool = False) -> None:
        """Run SVM. """

        X, y = self._integrate_train_data()

        print("Training params:")
        print(self.get_params())

        if standardize:
            X = self._standardize(X)
            print("Input data is standardized.")

        print("Training start.")
        self.svm.fit(X, y)
        print("Done! The SVM classifier is available for prediction. ")

    def finetune(self,
                 standardize: bool = False,
                 save: bool = True,
                 save_path: str = ".") -> dict:
        """Search for the best hyperparameters using grid search. Use 3-fold cross-validation as default.
        WARNING: standardize may cause a considerable reduction of searching speed, though as said in sklearn API. """

        if not save_path:
            raise Exception("ERROR: Invalid save path.")

        # Set testing scales
        param_grid = [
            {'C': np.logspace(0, 4, num=5), 'kernel': ['linear']},
            {'C': np.logspace(0, 4, num=5), 'gamma': np.logspace(-3, 0, num=4), 'kernel': ['rbf']},
            {'C': np.logspace(0, 4, num=5), 'gamma': ['scale', 'auto'], 'kernel': ['rbf']},
            {'C': np.logspace(0, 4, num=5), 'gamma': np.logspace(-3, 0, num=4), 'degree': [2, 3, 4],
             'kernel': ['poly']},
            {'C': np.logspace(0, 4, num=5), 'gamma': ['scale', 'auto'], 'degree': [2, 3, 4], 'kernel': ['poly']}
        ]

        X, y = self._integrate_train_data()

        print("Training params:")
        print(self.get_params())

        if standardize:
            X = self._standardize(X)
            print("Input data is standardized.")

        print("Grid searching start. \nPlease wait... this may take a while. ")
        # Using parallel computation by setting n_jobs=10
        cv = GridSearchCV(self.svm, param_grid, n_jobs=10, verbose=0).fit(X, y)
        if save:
            with open(os.path.join(save_path, "{}/svm_param_{}.txt".format(self.site_type, self.site_type)), 'w') as f:
                f.write("GRID SEARCH RESULTS\n")
                for k, v in cv.cv_results_.items():
                    f.write(str(k) + ':' + str(v) + '\n')
                f.write("\nBEST PARAMS FOR SVM\n")
                for k, v in cv.best_params_.items():
                    f.write(str(k) + ':' + str(v) + '\n')
        print("Done! Param scores saved in {}/{}/svm_param_{}.txt.".format(save_path, self.site_type, self.site_type))
        return cv.best_params_

    @staticmethod
    def _2dprob_to_1dprob(probs: np.ndarray) -> np.ndarray:

        p_ratio = [probs[i][0] / max(probs[i][1], 1e-6) for i in range(probs.shape[0])]
        return np.asarray([np.log(max(p, 1e-6)) for p in p_ratio])

    def predict(self,
                seq: Union[str, np.ndarray],
                threshold: Union[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:

        if isinstance(seq, str):
            # Sliding & Encoding
            length = len(seq)
            test_seqs = np.array([seq[i: i + self.feat_length] for i in range(0, length - self.feat_length)])
        else:
            test_seqs = seq
        X_test = []
        abandoned = [] # Save abandoned sites
        for i, site in enumerate(test_seqs):
            if len(self._remove_ambiguous_bases(site)) < self.feat_length:
                abandoned.append(i)
            else:
                X_test.append(self._encode_a_single_seq(site))

        if self.svm.probability:
            # Scoring using Platt scaling
            score = self._2dprob_to_1dprob(self.svm.predict_proba(np.asarray(X_test)))
        else:
            # Scoring using the decision function
            score = self.svm.decision_function(np.asarray(X_test))
            # TODO: This scoring method is incomplete. Complete this if doable.
        seq_pred = score > threshold
        return np.nonzero(seq_pred == True)[0] + self.data.up_sites, seq_pred, score, abandoned

    def save_model(self,
                   save_path: str = "../Model",
                   model_name: str = "SVM") -> None:

        if not save_path:
            raise Exception("ERROR: Invalid save path.")

        joblib.dump(filename=os.path.join(save_path, "{}/{}.model".format(self.site_type, model_name)), value=self.svm)
        print("Done! Model saved in {}/{}.model.".format(save_path, "{}/{}".format(self.site_type, model_name)))

    def load_model(self,
                   load_path: str = "../Model",
                   model_name: str = "SVM") -> None:

        if not os.path.join(load_path, "{}/{}.model".format(self.site_type, model_name)):
            raise Exception("ERROR: Invalid file name.")

        self.svm = joblib.load(os.path.join(load_path, "{}/{}.model".format(self.site_type, model_name)))
        print("Done! Model loaded from {}/{}.model.".format(load_path, "{}/{}".format(self.site_type, model_name)))


if __name__ == "__main__":

    import time

    TRAINING_PATH = "../Data/Training Set"
    TESTING_PATH = "../Data/Testing Set"

    # Set params

    site_type = "donor"
    model_path = ''  # Use '.' to activate load-model mode
    threshold = 2.4 # Best threshold

    seq = Sequence(filepath=TRAINING_PATH, type="train")
    test_seq = Sequence(filepath=TESTING_PATH, type="test")

    if model_path:
        svm = Svm(seq, site_type)
        svm.load_model(model_path, model_name="SVM")

    else:
        svm = Svm(seq, site_type)
        svm.encode()

        svm.set_params(kernel="linear", C=100., gamma='scale', cache_size=1000)
        # # Activate this and annotate other rows below for grid searching
        # best_params = svm.finetune(standardize=False)
        # print(best_params)
        svm.fit(standardize=False)

    svm.save_model(model_name="SVM")

    print("Predicting...")
    start = time.time()
    scores, preds, reals = svm.calc_scores(test_seq, threshold, testset_ratio=1)
    cm, r, p, fpr, tpr, f1 = svm.evaluate(preds, reals)
    tn, fn, fp, tp = cm.ravel()
    print("SVM: threshold={}, tn={}, fp={}, fn={}, tp={}, \nrecall={}, precision={}, fpr={}, tpr={}, f1={}".format(
        threshold, tn, fp, fn, tp, r, p, fpr, tpr, f1))

    end = time.time()
    print("SVM runtime: {}".format(end - start))

