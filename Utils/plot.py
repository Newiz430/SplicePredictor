#!/usr/bin/env python

from typing import List, Tuple, Any
import os
import argparse
from itertools import cycle, tee
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as isl
# from scipy.interpolate import make_interp_spline

from Utils.extract import Sequence
from Model.wam import Wmm, Wam
from Model.bn import BayesNet
from Model.svm import Svm

# Training settings (unused because it does not work as expected ?)
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--load_model', type=bool, default=False, help='Using pre-saved models (Default: False)')
args = parser.parse_args()


class Plot:

    def __init__(self,
                 trainset: Sequence,
                 site_type: str,
                 testset: Sequence,
                 save_path: str):
        """Plotting class for various curves.

        Parameters
        ----------
        trainset: A Utils.extract.Sequence instance
            Contains the signal sequences for model training.
            Donor & acceptor sites are saved in data.donor_site and data.acceptor_site,
            whose negative control sequences are in data.neg_donor_site and neg_acceptor_site.
            (The two neg site lists are the same by default.)

        site_type: str (one of "donor", "acceptor")
            Specifies the training sequences we will use to train the model.

        testset: A Utils.extract.Sequence instance
            Contains the testing sequences for evaluating.

        save_path: str or path_like
            The directory path for the pics.

        Examples
        --------
        See the main module.
        """

        if site_type not in {"donor", "acceptor"}:
            raise Exception("ERROR: Wrong site type.")

        self.trainset = trainset
        self.site_type = site_type
        self.testset = testset
        self.save_path = save_path

    def _fit(self,
             model: Any,
             kernel: str = "linear",
             load_model: bool = False,
             model_path: str = "../Model",
             save_model: bool = True) -> Any:

        if load_model:
            model.load_model(model_path, model_name=model.name)
        elif model.name in {"WMM", "WAM"}:
            model.fit()
        elif model.name == "SVM":
            model.encode()
            # Set parameters. Arguments below are the best param values obtained by grid searching. See Svm.finetune().
            if kernel == "linear":
                model.set_params(kernel=kernel, C=100., gamma='scale', cache_size=1000)
            elif kernel == "rbf":
                model.set_params(kernel=kernel, C=10., gamma='scale', cache_size=1000)
            else:
                model.set_params(kernel=kernel, C=1., gamma='scale', degree=4, cache_size=1000)
            model.fit(standardize=False)
        elif model.name == "BN":
            model.fit(struct_algorithm="PC", param_algorithm="MLE", alpha=1e-5)

        if save_model:
            model.save_model(save_path=model_path)

        return model

    def _get_metrics(self,
                     model_type: str,
                     data_path: str,
                     load_csv: bool,
                     load_model: bool,
                     save_metrics: bool = True) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:

        if model_type not in {"WMM", "WAM", "BN", "linear-SVM", "rbf-SVM", "poly-SVM"}:
            raise Exception("ERROR: Invalid model type.")

        if load_csv: # If there are csv files with data

            ap, auc = np.loadtxt(os.path.join(data_path, "{}/{}".format(self.site_type, model_type + ".csv")),
                                 delimiter=",", usecols=(6, 7), max_rows=1, unpack=True)
            thr, r, p, fpr, tpr = np.loadtxt(os.path.join(data_path, "{}/{}".format(self.site_type,
                                                                                    model_type + ".csv")),
                                             delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5), unpack=True)

        else:

            if model_type == "WMM":
                model = self._fit(Wmm(self.trainset, self.site_type), load_model=load_model)
            elif model_type == "WAM":
                model = self._fit(Wam(self.trainset, self.site_type), load_model=load_model)
            elif model_type == "BN":
                model = self._fit(BayesNet(self.trainset, self.site_type), load_model=load_model)
            elif model_type == "linear-SVM":
                model = self._fit(Svm(self.trainset, self.site_type), load_model=load_model, kernel="linear")
            elif model_type == "rbf-SVM":
                model = self._fit(Svm(self.trainset, self.site_type), load_model=load_model, kernel="rbf")
            else: # "poly-SVM"
                model = self._fit(Svm(self.trainset, self.site_type), load_model=load_model, kernel="poly")

            print("Calculating {} metrics by multiple thresholds...".format(model_type))
            scores, _, reals = model.calc_scores(self.testset, load_scores=False, save_scores=False)
            thr, r, p, ap, fpr, tpr, auc = model.get_plot_data(scores, reals)

            if save_metrics:
                f = open(os.path.join(data_path + "/{}".format(model.site_type), "{}.csv".format(model_type)), 'w')
                # Put AP, AUC at the end of the first line
                f.write("model,threshold,r,p,fpr,tpr,{},{}\n".format(ap, auc))
                for i in range(len(thr)):
                    f.write("{},{},{},{},{},{}\n".format(model_type, thr[i], r[i], p[i],
                                                         fpr[i], tpr[i]))
                f.close()
                print("Metrics of {} model is saved in {}/{}.csv.".format(
                    model_type, data_path + "/{}".format(model.site_type), model_type))

        return thr, r, p, ap, fpr, tpr, auc

    def _getdata(self,
                 model_types: List[str],
                 data_path: str,
                 load_csv: bool,
                 load_model: bool,
                 save_metrics: bool) -> List[Tuple]:

        data = [self._get_metrics(model, data_path, load_csv, load_model, save_metrics)
                for model in model_types]

        return data

    # Unused methods for smoothing the curves
    # @staticmethod
    # def _sort(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #
    #     ord = np.lexsort((y, x))
    #     return np.array(x[ord]), np.array(y[ord])

    # @staticmethod
    # def _interpolate(x: np.ndarray, y: np.ndarray, total_point: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    #
    #     # If there are duplicates in x
    #     x, idx = np.unique(x, return_index=True)
    #     y = y[idx] # Just take the first one
    #     # Interpolate
    #     x_s = np.linspace(min(x), max(x), total_point)
    #     y_s = make_interp_spline(x, y)(x_s)
    #
    #     return x_s, y_s

    def plot(self,
             model_types: List[str],
             load_csv: bool = False,
             load_model: bool = True,
             save_csv: bool = True,
             data_path: str = "../Model") -> None:
        """Plot P-R & ROC curves for models and save the figure in the given save_path. """

        if not data_path:
            raise Exception("ERROR: Invalid save path.")

        # Get metrics of all models included in model_types by multiple thresholds.
        # Rows in plot_data in order: threshold, recall, precision, AP, FPR, TPR, AUC
        plot_data = self._getdata(model_types, data_path, load_csv, load_model, save_csv)

        ls1, ls2, ls3, ls4 = tee(cycle(["-", "-.", "--", ":"]), 4)
        # Draw P-R curves
        pr, ax = plt.subplots(1, 1)
        for i, model_type in enumerate(model_types):
            # Set fixed points and plot
            if model_type == "WAM":
                r = np.concatenate(([0], plot_data[i][1], [1]))
                p = np.concatenate(([1], plot_data[i][2], [0]))
            elif model_type in {"WMM", "linear-SVM", "rbf-SVM", "poly-SVM"}:
                # Excise the last few erroneous points (created by sklearn.metrics.precision_recall_curve)
                r = np.concatenate(([0], plot_data[i][1][:-90], [1]))
                p = np.concatenate(([1], plot_data[i][2][:-90], [0]))
            else: # "BN"
                r = np.concatenate(([0], plot_data[i][1][:-2], [1]))
                p = np.concatenate(([1], plot_data[i][2][:-2], [0]))
            ax.step(r, p, where="post", ls=next(ls1),
                label="{} model (ap: {})".format(model_type, plot_data[i][3].round(4)))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="best", prop={'size' : 6})
        # # Make an inset (activate this for dense curves)
        # axins = isl.inset_axes(ax, width="30%", height="40%", loc=10)
        # for i, model_type in enumerate(model_types):
        #     if model_type == "WAM":
        #         r = np.concatenate(([0], plot_data[i][1], [1]))
        #         p = np.concatenate(([1], plot_data[i][2], [0]))
        #     elif model_type in {"WMM", "linear-SVM", "rbf-SVM", "poly-SVM"}:
        #         # Excise the last few erroneous points (created by sklearn.metrics.precision_recall_curve)
        #         r = np.concatenate(([0], plot_data[i][1][:-90], [1]))
        #         p = np.concatenate(([1], plot_data[i][2][:-90], [0]))
        #     else: # "BN"
        #         r = np.concatenate(([0], plot_data[i][1][:-2], [1]))
        #         p = np.concatenate(([1], plot_data[i][2][:-2], [0]))
        #     axins.step(r, p, where="post", ls=next(ls2),
        #         label="{} model (ap: {})".format(model_type, plot_data[i][3].round(4)))
        # axins.set_xlim(0.85, 0.95)
        # axins.set_ylim(0.9, 1.)
        # axins.xaxis.set_major_locator(plt.MultipleLocator(0.05))
        # axins.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        # isl.mark_inset(ax, axins, loc1=2, loc2=4, ls='--')
        plt.savefig(os.path.join(self.save_path, "prcurve.svg"))
        print("Done! PR curve saved in {}/prcurve.svg.".format(self.save_path))
        plt.close()

        # Draw ROC curves
        roc, ax = plt.subplots(1, 1)
        for i, model_type in enumerate(model_types):
            fpr = np.concatenate(([], plot_data[i][4], [1]))
            tpr = np.concatenate(([], plot_data[i][5], [1]))
            ax.plot(fpr, tpr, next(ls3),
                    label="{} model (auc: {})".format(model_type, plot_data[i][6].round(4)))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="lower right", prop={'size' : 6})
        # Make an inset (activate this for dense curves)
        axins = isl.inset_axes(ax, width="30%", height="40%", loc=10)
        for i, model_type in enumerate(model_types):
            fpr = np.concatenate(([], plot_data[i][4], [1]))
            tpr = np.concatenate(([], plot_data[i][5], [1]))
            axins.plot(fpr, tpr, next(ls4))
        axins.set_xlim(0., 0.05)
        axins.set_ylim(0.95, 1.)
        isl.mark_inset(ax, axins, loc1=1, loc2=3, ls='--')
        plt.savefig(os.path.join(self.save_path, "roccurve.svg"))
        print("Done! ROC curve saved in {}/roccurve.svg.".format(self.save_path))
        plt.close()


    def _plot_curve_with_maximum_point(self,
                                       thr: np.ndarray,
                                       r: np.ndarray,
                                       p: np.ndarray,
                                       f1: np.ndarray,
                                       pic_name: str) -> None:
        """Draw a curve and label the crest. """

        fig, ax = plt.subplots(1, 1)
        ax.plot(thr, p, '--', label="Precision")
        ax.plot(thr, r, '-.', label="Recall")
        ax.plot(thr, f1, ':', label="F1 score")
        # Annotate the peak f1 score
        f1_max = np.argmax(f1)
        max_coord = '(' + str(thr[f1_max].round(2)) + ' ' + str(f1[f1_max].round(2)) + ')'
        ax.plot(thr[f1_max], f1[f1_max], '.', color='g')
        ax.annotate(max_coord, xy=(thr[f1_max], f1[f1_max]), xytext=(thr[f1_max] - 1, f1[f1_max]), color='g')

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metrics")
        ax.legend(loc="best")
        plt.savefig(os.path.join(self.save_path, pic_name + ".svg"))
        plt.close()

    def plot_f1(self,
                model_type: str,
                load_csv: bool = True,
                data_path: str = "../Model") -> None:
        """Draw metrics-threshold curves and save the figure in the given save_path. """

        thr, r, p = self._getdata([model_type], data_path, load_csv, True, False)[0][:3]

        f1 = 2 * r * p / (r + p)
        self._plot_curve_with_maximum_point(thr, r, p, f1, "{}_threshold".format(model_type))

        print("Done! Threshold curves saved in {}/{}_threshold.svg.".format(self.save_path, model_type))
        plt.close()


if __name__ == "__main__":

    TRAINING_PATH = "../Data/Training Set"
    TESTING_PATH = "../Data/Testing Set"

    site_type = "donor"
    seq = Sequence(filepath=TRAINING_PATH, type="train")
    test_seq = Sequence(filepath=TESTING_PATH, type="test")

    plot = Plot(seq, site_type, test_seq, "../Pics")
    # Example of plotting P-R & ROC curves (you may need to adjust the terminal points of those annoying curves)
    plot.plot(["WMM", "WAM", "BN", "linear-SVM", "rbf-SVM", "poly-SVM"],
              load_csv=True, load_model=False, save_csv=False)
    # Example of finding thresholds
    plot.plot_f1("WMM")
