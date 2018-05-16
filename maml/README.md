This folder contains the code for the MAML benchmark. It requires the
same subdirectory structure as the root directory of this distribution -
specifically, the data/, ckpt/, and results/ subdirectories, with the
small datasets for testing included in data/.

The model is trained using trainMAML.py, and the table of results for
the benchmark is generated with test_datasetsMAML.py. Currently, these
must be merged by hand to generate the table in our paper.