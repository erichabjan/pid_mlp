
In this project we test the effectiveness of machine learning techniques to identify particles in the GlueX detector at Jefferson Laboratory. We trained multi-lyared percpetron (MLP) and boosted decision tree (BDT) models on smeared monte carlo data and compared these models to standard particle identification (PID) techniques.

Inside of the `mlp_pid` folder the MLP models of the charged and neutral particles are trained and saved using Tensorflow. The classification of particles is then carried out in the `mlp_pid.py` script where arrays of the outputs of this anlysis are saved to the `paper_plots` folder. All MLP results can be recreated in the `paper_plots.ipynb` notebook. 

As for manual PID, all analysis is carried out in the `manual_pid.py` script in the `manual_pid` folder. This includes both dE/dx - p and timing cuts, both of which are optimized in this notebook with the optimal dE/dx - p cuts being saved to the `paper_plots` folder along with the manual PID results. Similarly, all manual PID plots can be recreated in the `paper_plots.ipynb` notebook.

This code accompanies the paper: **"Particle identification in the GlueX detector using a multi-layer perceptron"** ([arXiv:2505.14706](https://arxiv.org/abs/2505.14706)). 