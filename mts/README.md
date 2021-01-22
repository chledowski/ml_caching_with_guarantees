Online metric algorithms with untrusted predictions
===================================================
Antonios Antoniadis, Christian Coester, Marek Elias, Adam Polak and Bertrand Simon

Source code accompanying paper https://arxiv.org/abs/2003.02144

Dependencies
------------

You need Python 3 with additional packages: numpy, matplotlib, and tqdm.
You also need pdflatex in order to produce the plots.

To install them using pip run:

sudo apt install python3-pip texlive-latex-base
pip3 install numpy matplotlib tqdm


Datasets
--------

We run our experiments on input instances obtained from publicly available
datasets Brightkite and CitiBike. The provided Bash script download_data.sh
downloads the datasets, extracts the instances, and puts them in a newly created
directory data. It may take about 10 minutes. For your convenience we provide
the ready instances in data.zip archive. You can simply unzip it instead of
running the script.


Experiments
-----------

To compare caching algorithms augmented with PLECO and POPU predictions run:

./main.py all -k 10 data/bk*.txt -n 10
./main.py all -k 100 data/citi*.txt -n 10

That is how we obtained numbers for Table 1.

To generate plots, i.e., BK dataset for caching (main paper and appendix), Citi
dataset for caching (appendix), Ice cream problem (main paper), run:


./main.py plot -k 10 data/bk*.txt -n 10 -o results/caching_bk_paper
./main.py plot -k 10 data/bk*.txt -n 10 -o results/caching_bk_appendix -s appendix -l results/caching_bk_paper.json 
./main.py plot -k 100 data/citi*.txt -n 10 -o results/caching_citi_appendix -s appendix
./icecream.py data/ic*.txt

The plots are generated both in png and pgf formats, and the raw data is stored
in json files. The results we obtained are stored in the results/ folder.
