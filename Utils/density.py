#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_density(scores: np.ndarray,
                 save_path: str = "../Pics",
                 pic_name: str = "density") -> None:
    """Draw a density picture for the scores under a certain threshold. """

    model = ["Donor", "Acceptor"]
    df = pd.DataFrame(scores.T, columns=model)
    # Plotting both distibutions on the same figure
    dens, ax = plt.subplots(1, 1)
    dens = sns.kdeplot(data=df, shade=True).get_figure()
    ax.set_xlabel("Score")
    dens.savefig(os.path.join(save_path, pic_name + ".svg"))
    print("Done! Density graph saved in {}/{}.svg.".format(save_path, pic_name))


if __name__ == "__main__":

    from Utils.extract import Sequence
    from Model.wam import Wam
    TRAINING_PATH = "../Data/Training Set"
    TESTING_PATH = "../Data/Testing Set"

    threshold = 0
    seq = Sequence(filepath=TRAINING_PATH, type="train")
    test_seq = Sequence(filepath=TESTING_PATH, type="test")

    wam_donor = Wam(seq, "donor")
    wam_donor.fit()
    wam_acceptor = Wam(seq, "acceptor")
    wam_acceptor.fit()

    donor_scores = wam_donor.calc_scores(test_seq, threshold, True, True)[0]
    acceptor_scores = wam_acceptor.calc_scores(test_seq, threshold, True, True)[0][0:len(donor_scores)]
    plot_density(np.vstack((donor_scores, acceptor_scores)))