# How-to-Make-the-Gradients-Small-Privately
Code for the experiments in "How to Make the Gradients Small Privately: Improved Rates for Differentially Private Non-Convex Optimization," (ICML 2024) by Andrew Lowy, Jonathan Ullman, and Stephen J. Wright. 

To (approximately) reproduce the experiments in the paper, follow these steps:
  1. Hyperparameter tuning: either use the hyperparameters in Appendix G of the paper or set "Tune == True" and "do_experiments == False" in the script to conduct tuning.
  2. Experiments: Set "Tune == False" and "do_experiments == True" in the script. Change the algorithmic parameters as needed so that they align with your desired hyperparameters. Run the script and save the printed results somewhere.

Due to randomness in the synthetic data generation process and in the noisy algorithms, results may differ from the ones reported in the paper. 
