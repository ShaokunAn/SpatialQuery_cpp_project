from typing import List, Optional

from anndata import AnnData
from spatial_module import SpatialDataMultiple


class spatial_query_multi:
    def __init__(self,
                 adatas: List[AnnData],
                 datasets: List[str],
                 spatial_key: str,
                 label_key: str,
                 leaf_size: int):
        self.spatial_key = spatial_key
        self.label_key = label_key
        # Modify dataset names by d_0, d_2, ... for duplicates in datasets
        count_dict = {}
        modified_datasets = []
        for dataset in datasets:
            if '_' in dataset:
                print(f"Warning: Misusage of underscore in '{dataset}'. Replacing with hyphen.")
                dataset = dataset.replace('_', '-')

            if dataset in count_dict:
                count_dict[dataset] += 1
            else:
                count_dict[dataset] = 0

            mod_dataset = f"{dataset}_{count_dict[dataset]}"
            modified_datasets.append(mod_dataset)

        self.datasets = modified_datasets

        self.spatial_query_multiple = SpatialDataMultiple()
        for i, adata in enumerate(adatas):
            self.spatial_query_multiple.set_fov_data(adata.obsm[self.spatial_key],
                                                     adata.obs[self.label_key])  # store data in cpp codes

    def build_fptree_knn(self,
                         ct: str,
                         dataset: Optional[str] = None,
                         ):
        # identify the datasets that match the given dataset name as well as containing the given cell type ct
        # intput indices of the matched datasets.
        pass

    def motif_enrichment_knn(self, motifs: Optional = None):
        if motifs is None:
            motifs = self.build_fptree_knn()
        else:
            # remove non-exist cell types in motifs
            pass
        out = self.spatial_query_multiple.motif_enrichment_knn(motifs)
        return out

    def find_fp_knn_fov(self,
                        ct: str,
                        dataset_id: int,
                        k: int=30,
                        min_support: float = 0.5,
                        ):
        pass

    def differential_analysis_knn(self,
                                  ct: str,
                                  datasets: List[str]):
        # Use same codes but use replace the frequent patterns search with cpp methods
        if len(datasets) != 2:
            raise ValueError("Require 2 datasets for differential analysis.")
        # Check if the two datasets are valid
        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in datasets:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n"
                                 f"Valid dataset names are: {set(valid_ds_names)}")

        flag = 0
        for d in datasets:
            fp_d = {}
            dataset_i = [ds for ds in self.datasets if ds.split('_')[0] == d]
            for d_i in dataset_i:
                fp_fov = self.find_fp_knn_fov(ct=ct,
                                              dataset_i=d_i,
                                              k=k,
                                              min_count=min_count,
                                              min_support=min_support)
                if len(fp_fov) > 0:
                    fp_d[d_i] = fp_fov

            if len(fp_d) == 1:
                common_patterns = list(fp_d.values())[0]
                common_patterns = common_patterns.rename(columns={'support': f"support_{list(fp_d.keys())[0]}"})
            else:
                comm_fps = set.intersection(*[set(df['itemsets']) for df in
                                              fp_d.values()])  # the items' order in patterns will not affect the returned intersection
                common_patterns = pd.DataFrame({'itemsets': list(comm_fps)})
                for data_name, df in fp_d.items():
                    support_dict = dict(df[['itemsets', 'support']].values)
                    support_dict = {tuple(key): value for key, value in support_dict.items()}
                    common_patterns[f"support_{data_name}"] = common_patterns['itemsets'].apply(
                        lambda x: support_dict.get(tuple(x), None))
            if flag == 0:
                fp_datasets = common_patterns
                flag = 1
            else:
                fp_datasets = fp_datasets.merge(common_patterns, how='outer', on='itemsets', ).fillna(0)

        match_ind_datasets = [
            [col for ind, col in enumerate(fp_datasets.columns) if col.startswith(f"support_{dataset}")] for dataset in
            datasets]
        p_values = []
        dataset_higher_ranks = []
        for index, row in fp_datasets.iterrows():
            group1 = pd.to_numeric(row[match_ind_datasets[0]].values)
            group2 = pd.to_numeric(row[match_ind_datasets[1]].values)

            # Perform the Mann-Whitney U test
            stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided', method='auto')
            p_values.append(p)

            # Label the dataset with higher frequency of patterns based on rank median
            support_rank = pd.concat([pd.DataFrame(group1), pd.DataFrame(group2)]).rank()  # ascending
            median_rank1 = support_rank[:len(group1)].median()[0]
            median_rank2 = support_rank[len(group1):].median()[0]
            if median_rank1 > median_rank2:
                dataset_higher_ranks.append(datasets[0])
            else:
                dataset_higher_ranks.append(datasets[1])

        fp_datasets['dataset_higher_frequency'] = dataset_higher_ranks
        # Apply Benjamini-Hochberg correction for multiple testing problems
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')

        # Add the corrected p-values back to the DataFrame (optional)
        fp_datasets['corrected_p_values'] = corrected_p_values
        fp_datasets['if_significant'] = if_rejected

        # Return the significant patterns in each dataset
        fp_dataset0 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[0]) & (fp_datasets['if_significant'])
            ][['itemsets', 'corrected_p_values']]
        fp_dataset1 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[1]) & (fp_datasets['if_significant'])
            ][['itemsets', 'corrected_p_values']]
        fp_dataset0 = fp_dataset0.reset_index(drop=True)
        fp_dataset1 = fp_dataset1.reset_index(drop=True)
        fp_dataset0 = fp_dataset0.sort_values(by='corrected_p_values', ascending=True)
        fp_dataset1 = fp_dataset1.sort_values(by='corrected_p_values', ascending=True)
        return fp_dataset0, fp_dataset1
