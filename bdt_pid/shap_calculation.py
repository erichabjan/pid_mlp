import xgboost as xgb
import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model, data_to_explain, background_data, classes = [], n_background_per_particle=200, n_explain_per_particle=1000):
        self.model = model
        self.classes = classes
        self.feature_names = data_to_explain.columns.tolist()
        for feature_to_remove in ['ptype','true ptype','group']:
            if feature_to_remove in self.feature_names:
                self.feature_names.remove(feature_to_remove)
        self.nclasses = len(classes)
        temp_data_to_explain = data_to_explain.sample(frac=1).drop_duplicates('group').drop(columns=['true ptype','ptype','group']).reset_index(drop=True)
        self.data_to_explain = temp_data_to_explain[temp_data_to_explain.index % 20000 < n_explain_per_particle].reset_index(drop=True).to_numpy()

        if isinstance(self.model, tf.keras.Model):
            self.model_type = "MLP"
            self.explainer = shap.DeepExplainer(model, self.background_data)
            self.background_data = background_data[background_data.index % 80000 < n_background_per_particle].drop(columns=['ptype']).reset_index(drop=True).to_numpy()
        elif isinstance(self.model, xgb.Booster):
            self.model_type = "BDT"
            self.explainer = shap.TreeExplainer(model)
        self.shap_values=None
    def calculate_shap(self):
        self.shap_values = self.explainer.shap_values(self.data_to_explain)
        self.shap_values = np.transpose(self.shap_values, (2, 0, 1))
    def plot_summary(self, max_display=20):

        if self.shap_values is None:
            raise RuntimeError("Run calculate_shap() before plotting.")
        for i in range(self.nclasses):
            plt.figure()
            shap.plots.violin(
                self.shap_values[i],
                features=self.data_to_explain,
                feature_names=self.feature_names,
                max_display=max_display,
                plot_type='layered_violin',
                plot_size=(6.4,4.8)
            )
            plt.title(self.model_type + " SHAP Summary Plot for " + self.classes[i])
            plt.tight_layout()
            plt.xlabel("SHAP Value")
            ax = plt.gca()
            plt.setp(ax.get_yticklabels(), fontsize=10)
            plt.savefig(f"/home/rdube/PID_paper/pid_mlp/bdt_pid/shap_summary_class_{self.classes[i]}.png", dpi=300)
            plt.close()

    def get_mean_shap(self):
        return np.array([np.abs(s).mean(axis=0) for s in self.shap_values])
    
    def save_shap_values(self,path):
        np.save(path, self.shap_values)


        
        




