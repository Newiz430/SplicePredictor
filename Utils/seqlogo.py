#!/usr/bin/env python

import os
from typing import List

import weblogo.logo as logo
from weblogo import SeqList, Alphabet, LogoData, LogoOptions, LogoFormat, logo_formatter
from weblogo.seq import unambiguous_dna_alphabet


def draw_seqlogo(seqs: List[str],
                 save_path: str =  "../Pics",
                 pic_name: str = "logo",
                 pic_type: str = "png",
                 alphabet: Alphabet = unambiguous_dna_alphabet,
                 resolution: float = 1000.) -> None:
    """Draw a WebLogo pic.
    Requires the interpreter Ghostscript which is available at https://www.ghostscript.com/download/gsdnld.html.
    .svg format pic requires pdf2svg which is available at https://github.com/dawbarton/pdf2svg.
    Choose alphabet = unambiguous_dna_alphabet for nucleotide sequences. """

    if not save_path:
        raise Exception("ERROR: Invalid save path.")

    # Read sequences
    data = SeqList(seqs, alphabet)
    data = LogoData.from_seqs(data)
    # Set seqlogo options
    logooptions = LogoOptions()
    logooptions.show_fineprint = False
    logooptions.logo_title = "{} site".format(pic_name)
    logooptions.title_fontsize = 12
    logooptions.scale_width = False
    logooptions.stack_aspect_ratio = 10
    logooptions.stack_width = logo.std_sizes["large"]
    logooptions.resolution = resolution
    # Export image bytes
    logoformat = LogoFormat(data, logooptions)

    if pic_type == "png":
        formatter = logo_formatter.png_formatter(data, logoformat)
        with open(os.path.join(save_path, pic_name + ".png"), 'wb') as f:
            f.write(formatter)
    elif pic_type == "jpg":
        formatter = logo_formatter.jpeg_formatter(data, logoformat)
        with open(os.path.join(save_path, pic_name + ".jpg"), 'wb') as f:
            f.write(formatter)
    elif pic_type == "svg":
        formatter = logo_formatter.svg_formatter(data, logoformat)
        with open(os.path.join(save_path, pic_name + ".svg"), 'wb') as f:
            f.write(formatter)
    else:
        raise Exception("ERROR: Unsupported picture extension. ")

    print("Done! Logo pic saved in Pics/{}.{}.".format(pic_name, pic_type))


if __name__ == "__main__":

    from Utils.extract import Sequence
    TRAINING_PATH = "../Data/Training Set"

    seq = Sequence(filepath=TRAINING_PATH)
    draw_seqlogo(seq.donor_site, pic_name="Donor")
    draw_seqlogo(seq.acceptor_site, pic_name="Acceptor")