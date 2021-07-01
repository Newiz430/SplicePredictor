#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def single_site_heatmap(p_mat: np.ndarray,
                        up_sites: int,
                        down_sites: int,
                        verbose: bool = True,
                        save_path: str = "../Pics",
                        pic_name: str = "ssheatmap") -> None:
    """Draw a single-site distribution heatmap using Seaborn. """

    if not save_path:
        raise Exception("ERROR: Invalid save path.")

    # Generate picture
    sites = list(map(str, range(- up_sites, 0))) + ["site", "site"] + list(map(str, range(1, down_sites + 1)))
    bases = ('A', 'C', 'G', 'T')
    df = pd.DataFrame(p_mat, index=sites, columns=bases)
    # Set options
    annot = True if verbose else None
    annot_kws = {"size": 7} if verbose else None
    ssheat = sns.heatmap(df, cmap="YlGnBu", annot=annot, annot_kws=annot_kws, vmax=1, vmin=0).get_figure()
    ssheat.savefig(os.path.join(save_path, pic_name + ".svg"), dpi=300)
    print("Done! Heatmap saved in {}/{}.png.".format(save_path, pic_name))
    plt.close()


def adjacent_site_heatmap(p_mat: np.ndarray,
                          up_sites: int,
                          down_sites: int,
                          verbose: bool = True,
                          save_path: str = "../Pics",
                          pic_name: str = "asheatmap") -> None:
    """Draw an adjacent-site distribution heatmap using Seaborn. """

    if not save_path:
        raise Exception("ERROR: Invalid save path.")

    p_mat = np.resize(p_mat, (p_mat.shape[0], 16, )) # Flatten the 3-D matrix to create 2-D heatmap
    sites = list(map(str, range(- up_sites, 0))) + ["site", "site"] + list(map(str, range(1, down_sites)))
    bases = ("A-A", "A-C", "A-G", "A-T", "C-A", "C-C", "C-G", "C-T",
             "G-A", "G-C", "G-G", "G-T", "T-A", "T-C", "T-G", "T-T")
    df = pd.DataFrame(p_mat, index=sites, columns=bases)
    # Set options
    annot = True if verbose else None
    annot_kws = {"size": 5} if verbose else None
    asheat = sns.heatmap(df, cmap="YlGnBu", annot=annot, annot_kws=annot_kws, vmax=1, vmin=0).get_figure()
    asheat.savefig(os.path.join(save_path, pic_name + ".svg"), dpi=300)
    print("Done! Heatmap saved in {}/{}.svg.".format(save_path, pic_name))
    plt.close()


if __name__ == "__main__":

    from Utils.extract import Sequence
    from Model.wam import Wmm, Wam
    TRAINING_PATH = "../Data/Training Set"

    print("Reading sequence files...")
    seq = Sequence(filepath=TRAINING_PATH, up_sites=5, down_sites=5)

    wmm = Wmm(seq, "donor")
    wmm.fit()
    wam = Wam(seq, "donor")
    wam.fit()
    single_site_heatmap(wmm.p_minus, 5, 5)
    adjacent_site_heatmap(wam.p_minus, 5, 5)