import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from packages.imbalance_degree import imbalance_degree
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

class FairnessReport():
    
    def __init__(self, dataset, df, df_norm):
        """
        Attributes:
        dataset = raw dataset obtained from openML platform with fetch_openml method
        df - dataframe containing cleaned raw data (e.g., continuous variables set into intervals and set astype objects)
        df_norm - normalized dataframe
        """
        self.dataset = dataset
        self.df = df
        self.data = df_norm
    
    def calculate_entropy(self, series):
        unique_values = pd.unique(series)
        num_values = len(unique_values)
        
        if num_values == 1:
            return 0  # If there's only one unique value, entropy is 0
        
        if series.dtype == np.number:
            # Numerical attribute
            sorted_values = np.sort(unique_values)
            bins = [(sorted_values[i] + sorted_values[i+1]) / 2 for i in range(num_values - 1)]
            hist, _ = np.histogram(series, bins=bins)
            prob_distribution = hist / len(series)
        else:
            # Categorical attribute
            value_counts = series.value_counts()
            prob_distribution = value_counts / len(series)
        
        entropy_value = entropy(prob_distribution, base=num_values)
        return entropy_value

    def calculate_imbalance_ratio(self, series):
        if (series.dtype == np.object or series.dtype == 'category'):
            # For both binary & multi-class categorical attributes (for multi-class the results are "low-resolution")
            class_counts = series.value_counts()
            return class_counts.max() / class_counts.min()
        else:
            return np.nan
        
    def calculate_imbalance_degree(self, series, distance="EU"):
        if (series.dtype == np.object or series.dtype == 'category'):
            return imbalance_degree(series, distance)
        else:
            return np.nan
        
    def get_minority_classes(self, series):
        """
        Minority classes are considered to be those with lower empirical_distribution 
        in comparisson to related attribute equiprobability
        """
        if (series.dtype == np.object or series.dtype == 'category'):
            unique_classes, class_counts = np.unique(series, return_counts=True)
            empirical_distribution = class_counts / class_counts.sum()
            
            eqp = 1 / len(class_counts) # equiprobability
            
            result = {unique_classes[i]:x for i, x in enumerate(empirical_distribution) if x < eqp}
            result = dict(sorted(result.items(), key=lambda item: item[1]))
            return list(result.keys())
        else:
            return np.nan
        
    def get_majority_classes(self, series):
        """
        Majority classes are considered to be those with higher empirical_distribution 
        in comparisson to related attribute equiprobability
        """
        if (series.dtype == np.object or series.dtype == 'category'):
            unique_classes, class_counts = np.unique(series, return_counts=True)
            empirical_distribution = class_counts / class_counts.sum()
            
            eqp = 1 / len(class_counts) # equiprobability
            
            result = {unique_classes[i]:x for i, x in enumerate(empirical_distribution) if x > eqp}
            result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
            return list(result.keys())        
        else:
            return np.nan
        
    def get_attribute_label_encoder_mapping(self, attribute):
        le = LabelEncoder()
        le.fit(self.df[attribute])
        return dict(zip(le.classes_, le.transform(le.classes_))) 

    def detect_protected_attributes(self, dataframe, favorable_label, unfavorable_label):
        stats = {}
        concentrations = [10000, 5000, 1000, 100, 10, 1]
        while(len(concentrations)):
            finished = True
            for attr in dataframe.columns:
                entropy_val = self.calculate_entropy(dataframe[attr])
                imbalance_ratio = self.calculate_imbalance_ratio(dataframe[attr])
                imbalance_degree = self.calculate_imbalance_degree(dataframe[attr])
                majority_classes = self.get_majority_classes(self.df[attr])   
                minority_classes = self.get_minority_classes(self.df[attr])

                base_rate_privileged_all = np.nan
                base_rate_unprivileged_all = np.nan
                statistical_parity_difference_all = np.nan
                disparate_impact_ratio_all = np.nan
                sedf = np.nan
                if (dataframe[attr].dtype == np.object or dataframe[attr].dtype == 'category'):
                    label_names = self.dataset.target_names
                    if len(label_names) == 0:
                        label_names = [self.data.columns[-1]]

                    bld = BinaryLabelDataset(
                        df=self.data,
                        label_names=label_names,
                        protected_attribute_names=[attr],
                        favorable_label=favorable_label,
                        unfavorable_label=unfavorable_label)

                    le_name_mapping = self.get_attribute_label_encoder_mapping(attr)            
                    privileged_groups = [{attr: le_name_mapping[v]} for v in majority_classes]
                    unprivileged_groups = [{attr: le_name_mapping[v]} for v in minority_classes]

                    metric = BinaryLabelDatasetMetric(
                            bld, 
                            privileged_groups=privileged_groups,
                            unprivileged_groups=unprivileged_groups)
                            
                    base_rate_privileged_all = metric.base_rate(True)
                    base_rate_unprivileged_all = metric.base_rate(False)
                    statistical_parity_difference_all = metric.statistical_parity_difference()
                    disparate_impact_ratio_all = metric.disparate_impact()
                    sedf = metric.smoothed_empirical_differential_fairness(concentrations[-1])

                    if sedf > 1:
                        finished = False
                        concentrations.pop()
                        break

                if imbalance_degree is np.nan:
                    total_classes = np.nan
                else:
                    total_classes = len(pd.unique(dataframe[attr]))

                stats[attr] = {
                    'majority_classes': majority_classes,
                    'minority_classes': minority_classes,
                    'total_classes': total_classes,
                    'entropy': entropy_val,
                    'imbalance_ratio': imbalance_ratio,
                    'imbalance_degree': imbalance_degree,
                    'statistical_parity_difference': statistical_parity_difference_all,
                    'disparate_impact_ratio': disparate_impact_ratio_all,
                    'smoothed_edf': sedf
                }
            
            if finished:
                break
                
        #print('Concentration parameter for Dirichlet smoothing: ' + str(concentrations[-1]))
        return pd.DataFrame(stats).transpose()