#!/usr/bin/env python

import warnings
from typing import Any, List, Tuple, Union
import os
import time
from random import sample
import joblib
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import PC, MaximumLikelihoodEstimator, BayesianEstimator, HillClimbSearch, MmhcEstimator

from Model.basemodel import Base
from Utils.extract import Sequence

warnings.filterwarnings("ignore")


class BayesNet(Base):

    def __init__(self,
                 data: Sequence,
                 site_type: str) -> None:
        """A Bayesian Network for splice site prediction. """

        super(BayesNet, self).__init__(data, site_type)

        self.name = "BN"

        self.lnode = self.data.down_sites + 2 + 1
        self.nodes = list(map(str, np.concatenate((np.arange(- self.data.up_sites, 0), np.arange(1, self.lnode)))))
        self.df = self._pack_data()
        self.state_names = self._set_state_names()

        self.skel = None
        self.pdag = None
        self.dag = None
        self.bn = None

    def _set_state_names(self) -> dict:

        state_names = {site: list('ACGT') for site in self.df.columns[:-1]}
        state_names[self.df.columns[-1]] = [True, False]
        return state_names

    def _extract_site(self,
                      sites: List[str]) -> List[str]:

        for site in sites:
            if len(self._remove_ambiguous_bases(site)) == self.feat_length:
                yield list(site.upper())

    def _pack_data(self) -> pd.DataFrame:
        """Pack the splice site sequences into a pd.DataFrame. """

        if self.site_type == "donor":
            pos_sites = [sl for sl in self._extract_site(self.data.donor_site)] * 2
            neg_sites = sample([sl for sl in self._extract_site(self.data.neg_donor_site)], 5000)
        else:
            pos_sites = [sl for sl in self._extract_site(self.data.acceptor_site)] * 2
            neg_sites = sample([sl for sl in self._extract_site(self.data.neg_acceptor_site)], 5000)

        df = pd.DataFrame(np.concatenate((pos_sites, neg_sites), axis=0), columns=self.nodes)
        df['Sp'] = np.repeat((True, False), (len(pos_sites), len(neg_sites)))

        print("Sequences loaded successfully with {} * positive sequences and {} * negative sequences."
              .format(len(pos_sites), len(neg_sites)))
        return df

    def _learn_struct(self,
                      struct_algorithm: str = "PC",
                      alpha: float = 0.01,
                      verbose: bool = True
                      ) -> List[Tuple[int, int]]:
        """Learn the framework of DAG based on given data. This may take several hours to run. """

        if struct_algorithm not in {"PC", "Hill-Climb", "Mmhc"}:
            raise Exception("ERROR: Wrong structure learning algorithm.")

        print("Structure learning start. Using {} as the learning algorithm. ".format(struct_algorithm))

        if struct_algorithm == "PC":

            est = PC(self.df)
            # Generate the dependency graph (skel) and DAG
            self.skel, separating_sets = est.build_skeleton(significance_level=alpha, variant="parallel",
                                                            show_progress=int(verbose))
            if verbose:
                print("skeleton edges: \n", self.skel.edges)
            self.pdag = est.skeleton_to_pdag(self.skel, separating_sets)
            self.dag = self.pdag.to_dag()

        elif struct_algorithm == "Hill-Climb":

            est = HillClimbSearch(self.df, use_cache=True)
            # Using BDeu to score hill-climbing
            self.dag = est.estimate(scoring_method='k2score')

        else: # struct_algorithm == "Mmhc"

            est = MmhcEstimator(self.df)
            self.dag = est.estimate(scoring_method='k2score')

        # Adjust the arrow direction
        dag_edges = [(v, u) if u == 'Sp' else (u, v) for (u, v) in self.dag.edges]
        self.dag.clear_edges()
        self.dag.add_edges_from(dag_edges)

        if verbose:
            print("Edges of DAG: \n", self.dag.edges)

        return list(self.dag.edges)

    def fit(self,
            struct_algorithm: str = "PC",
            param_algorithm: str = "MLE",
            alpha: float = 0.01) -> Any:
        """Learn a Bayesian Network based on given data. """

        param_algorithms = {"MLE": MaximumLikelihoodEstimator, "Bayes": BayesianEstimator}

        start = time.time()
        self.bn = BayesianModel(ebunch=self._learn_struct(struct_algorithm, alpha))
        print("Parameter learning start. Using {} as the learning algorithm. ".format(param_algorithm))
        self.bn.fit(self.df, estimator=param_algorithms[param_algorithm], state_names=self.state_names)

        end = time.time()
        print("Done! You may use the network to predict your data now. (Learning runtime: {} s)".format(end - start))

        return self.bn

    def get_cpd_info(self):
        """Display CPDs of all nodes in the Bayesian Model. """

        for cpd in self.bn.get_cpds()[:-1]:
            print(cpd)
        label = self.bn.get_cpds()[-1]
        print("label CPD size: ", label.cardinality)
        print("relevant nodes: ", label.variables[1:])

    @staticmethod
    def _2dprob_to_1dprob(probs: np.ndarray) -> np.ndarray:

        # A lot of [0.5, 0.5] in negative predicting. They are changed to [0, 1] for a better P-R curve.
        p_ratio = [0 if probs[i][0] == probs[i][1] else probs[i][0] / max(probs[i][1], 1e-6)
                   for i in range(probs.shape[0])]
        return np.asarray([np.log(max(p, 1e-6)) for p in p_ratio])

    def predict(self,
                seq: Union[str, np.ndarray],
                threshold: Union[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:

        if isinstance(seq, str):
            # Sliding & Encoding
            length = len(seq)
            test_seqs = np.array([seq.upper()[i: i + self.feat_length] for i in range(0, length - self.feat_length)])
        else:
            test_seqs = seq
        abandoned = [] # Save abandoned sites
        for i, site in enumerate(test_seqs):
            if len(self._remove_ambiguous_bases(site)) < self.feat_length:
                abandoned.append(i)

        # Make the data frame
        X_test = pd.DataFrame([list(site) for site in np.delete(test_seqs, abandoned)], columns=self.nodes)
        X_test.drop(self.state_names - self.bn.nodes, axis=1, inplace=True)  # Remove latent sites

        score = self._2dprob_to_1dprob(self.bn.predict_probability(X_test).to_numpy())
        seq_pred = score > threshold
        return np.nonzero(seq_pred == True)[0] + self.data.up_sites, seq_pred, score, abandoned

    def copy(self) -> Any:
        """Create a clone for BN model. """

        clone = BayesNet(self.data, self.site_type)
        clone.bn = self.bn.copy() if self.bn else None
        return clone

    def save_model(self,
                   save_path: str = "../Model",
                   model_name: str = "BN") -> None:

        if not save_path:
            raise Exception("ERROR: Invalid save path.")

        value = (self.skel, self.pdag, self.dag, self.bn)
        joblib.dump(filename=os.path.join(save_path, "{}/{}.model".format(self.site_type, model_name)), value=value)
        print("Done! Model saved in {}/{}.model.".format(save_path, "{}/{}".format(self.site_type, model_name)))

    def load_model(self,
                   load_path: str = "../Model",
                   model_name: str = "BN") -> None:

        if not os.path.join(load_path, "{}/{}.model".format(self.site_type, model_name)):
            raise Exception("ERROR: Invalid file name.")

        model = joblib.load(os.path.join(load_path, "{}/{}.model".format(self.site_type, model_name)))
        self.skel, self.pdag, self.dag, self.bn = model
        print("Done! Model loaded from {}/{}.model.".format(load_path, "{}/{}".format(self.site_type, model_name)))


if __name__ == "__main__":

    TRAINING_PATH = "../Data/Training Set"
    TESTING_PATH = "../Data/Testing Set"

    # Set params

    site_type = "donor"
    threshold = 3.38 # Best threshold
    model_path = '.'  # Use '.' to activate load-model mode

    seq = Sequence(filepath=TRAINING_PATH, type="train")
    test_seq = Sequence(filepath=TESTING_PATH, type="test")

    bn = BayesNet(seq, site_type)
    if model_path:
        bn.load_model(model_path)
    else:
        bn.fit(struct_algorithm="PC", param_algorithm="MLE", alpha=1e-5)

    # print("1:\n",bn.bn.get_cpds(node='1').values)
    # print("2:\n",bn.bn.get_cpds(node='2').values)

    bn.save_model(model_name="BN")

    print("Predicting...")
    start = time.time()
    scores, preds, reals = bn.calc_scores(test_seq, threshold)
    cm, r, p, fpr, tpr, f1 = bn.evaluate(preds, reals)
    tn, fn, fp, tp = cm.ravel()
    print("BN: threshold={}, tn={}, fp={}, fn={}, tp={}, \nrecall={}, precision={}, fpr={}, tpr={}, f1={}".format(
        threshold, tn, fp, fn, tp, r, p, fpr, tpr, f1))

    end = time.time()
    print("BN runtime: {}".format(end - start))
