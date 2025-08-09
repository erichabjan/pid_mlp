import numpy as np
from misc import SHAP_plots, load_file

### Making SHAP Plots
neutral_ptype = [22,130,2112]
test_neutral = load_file("neutral","Test", neutral_ptype.index)

charged_ptype = {2212:0, 321:1, -13:2, 211:2, -11:3, -2212:4,-321:5,13:6, -211:6,11:7}
test_charged = load_file("charged","Test", charged_ptype)

repo_dirc = '/home/rdube/PID_paper/pid_mlp/'

# Load BDT charged arrays
charged_indices_bdt = np.load(repo_dirc + "bdt_plots/charged_indices_to_test.npy")
charged_data_to_test_bdt = test_charged.loc[charged_indices_bdt].drop(columns=['group','ptype','true ptype'])
charged_shaps_bdt = np.load(repo_dirc + "bdt_plots/charged_shap.npy")

# Load MLP charged arrays
charged_indices_mlp = np.load(repo_dirc + "paper_plots/charged_indices_to_test.npy")
charged_data_to_test_mlp = test_charged.loc[charged_indices_mlp].drop(columns=['group','ptype','true ptype'])
charged_shaps_mlp = np.load(repo_dirc + "paper_plots/charged_shap.npy")
charged_data_to_test_mlp


SHAP_plots(charged_shaps_mlp, charged_data_to_test_mlp, charged_shaps_bdt, charged_data_to_test_bdt, "pos")
SHAP_plots(charged_shaps_mlp, charged_data_to_test_mlp, charged_shaps_bdt, charged_data_to_test_bdt, "neg")

# Load BDT neutralarrays
neutral_indices_bdt = np.load(repo_dirc + "bdt_plots/neutral_indices_to_test.npy")
neutral_data_to_test_bdt = test_neutral.loc[neutral_indices_bdt].drop(columns=['group','ptype','true ptype'])
neutral_shaps_bdt = np.load(repo_dirc + "bdt_plots/neutral_shap.npy")

# Load MLP neutral arrays
neutral_indices_mlp = np.load(repo_dirc + "paper_plots/neutral_indices_to_test.npy")
neutral_data_to_test_mlp = test_neutral.loc[neutral_indices_mlp].drop(columns=['group','ptype','true ptype'])
neutral_shaps_mlp = np.load(repo_dirc + "paper_plots/neutral_shap.npy")

SHAP_plots(neutral_shaps_mlp, neutral_data_to_test_mlp, neutral_shaps_bdt, neutral_data_to_test_bdt, "neutral")