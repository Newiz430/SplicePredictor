#!/usr/bin/env python

from typing import List, Tuple
import os
import re

TRAINING_PATH = "../Data/Training Set"
TESTING_PATH = "../Data/Testing Set"


class Sequence:

    def __init__(self,
                 filepath: str,
                 type: str = "train",
                 up_sites: int = 5,
                 down_sites: int = 5) -> None:
        """All kinds of splice site sequences for training.

        Parameters
        ----------
        filepath: str or path-like
            The directory path of the dataset. Files in which are txts containing one sequence each.

        type: str (one of "train", "test")
            Set the data type. Way of reading would be different.

        up_sites: int, default 5
            Decide how many sites upstream of the label would be selected as feature sequences.

        down_sites: int, default 5
            Decide how many sites downstream of the label would be selected as feature sequences.

        Examples
        --------
        See the main module.
        """

        if not filepath:
            raise Exception("ERROR: Invalid dataset path.")

        self.type = type
        self.filepath = filepath
        self.up_sites = up_sites
        self.down_sites = down_sites
        self.donor_site = []
        self.acceptor_site = []
        self.neg_donor_site = []
        self.neg_acceptor_site = []

        if self.type not in {"train", "test"}:
            raise Exception("ERROR: Invalid data type.")

        self.extract()

    def read_seq(self,
                 filename: str) -> Tuple[str, int, List[int], List[int]]:
        """Reading a single txt file."""

        if not os.path.join(self.filepath, filename):
            raise Exception("ERROR: Invalid data file.")

        seq = ""
        with open(os.path.join(self.filepath, filename)) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    info = line
                if i == 1:
                    donor = list(map(int, re.findall("(?<=\.\.)\d+(?=,)", line)))
                    acceptor = list(map(int, re.findall("(?<=,)\d+(?=\.\.)", line)))
                if i > 1:
                    seq += line.strip()

        length = int(re.search("\d+(?= bp)", info).group(0)) if self.type == "train" else len(seq)

        return seq, length, donor, acceptor

    def extract_single_txt(self,
                           txt: str):
        """Extract sequence data in a single txt. """
        seq, length, donor, acceptor = self.read_seq(filename=txt)

        # Extract splice sites
        donor_site = [seq[fig - self.up_sites: fig + self.down_sites + 2] for fig in donor]
        acceptor_site = [seq[fig - self.up_sites - 3: fig + self.down_sites - 1] for fig in acceptor]

        # Generate negative sites
        neg = range(self.up_sites, length - self.down_sites - 2, self.up_sites + self.down_sites + 2)
        sites = []
        for fig in donor + acceptor:
            sites.extend(range(fig - (self.up_sites + self.down_sites),
                               fig + (self.up_sites + self.down_sites + 2)))
        neg = list(set(neg).difference(set(sites)))

        neg_donor = neg
        neg_acceptor = neg
        neg_donor_site = [seq[fig - self.up_sites: fig + self.down_sites + 2] for fig in neg_donor]
        neg_acceptor_site = [seq[fig - self.up_sites: fig + self.down_sites + 2] for fig in neg_acceptor]

        return donor_site, neg_donor_site, acceptor_site, neg_acceptor_site

    def extract(self) -> None:
        """Extract useful sites from the given dataset."""

        for txt in os.listdir(self.filepath):
            seq, length, donor, acceptor = self.read_seq(filename=txt)

            # Extract splice sites
            self.donor_site.extend([seq[fig - self.up_sites: fig + self.down_sites + 2] for fig in donor])
            self.acceptor_site.extend([seq[fig - self.up_sites - 3: fig + self.down_sites - 1]
                                       for fig in acceptor])

            if self.type == "train":
                # Generate negative sites
                neg = range(self.up_sites, length - self.down_sites - 2, self.up_sites + self.down_sites + 2)
                sites = []
                for fig in donor + acceptor:
                    sites.extend(range(fig - (self.up_sites + self.down_sites),
                                       fig + (self.up_sites + self.down_sites + 2)))
                neg = list(set(neg).difference(set(sites)))

                neg_donor = neg
                neg_acceptor = neg
                self.neg_donor_site.extend(seq[fig - self.up_sites: fig + self.down_sites + 2] for fig in neg_donor)
                self.neg_acceptor_site.extend(seq[fig - self.up_sites: fig + self.down_sites + 2]
                                              for fig in neg_acceptor)


if __name__ == "__main__":

    print("Reading sequence files...")
    seq = Sequence(filepath=TRAINING_PATH)
    seqs = Sequence(filepath=TESTING_PATH, type="test")
    seq.extract()
    for txt in os.listdir(seqs.filepath):
        seqs.extract_single_txt(txt)