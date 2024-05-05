from typing import List, Optional, Union

import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as mt
from anndata import AnnData
from scipy.stats import hypergeom
from spatial_module import SpatialDataMultiple
import time


class spatial_query_multiple:
    def __init__(self,
                 adatas: List[AnnData],
                 datasets: List[str],
                 spatial_key: str,
                 label_key: str,
                 ):
        self.spatial_key = spatial_key
        self.label_key = label_key

        # Set dataset names
        # Number dataset names as d_0, d_2, ... for multiple FOVs from each dataset
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

            modified_dataset = f"{dataset}_{count_dict[dataset]}"
            modified_datasets.append(modified_dataset)

        self.datasets = modified_datasets

        # Store labels of all FOVs
        labels_all = []
        for adata in adatas:
            labels_adata = adata.obs[self.label_key]
            labels_adata = labels_adata.astype('category')
            labels_adata = [label.replace("_", "-") for label in
                            labels_adata]  # we'll use "_" as delimiter in following analysis, so replace "_" with "-"
            labels_all.append(labels_adata)
        self.labels = labels_all

        # Set spatial coordinates and build KDTree in cpp object
        self.spatial_query_multiple = SpatialDataMultiple()
        for i, adata in enumerate(adatas):
            self.spatial_query_multiple.set_fov_data(
                i,
                adata.obsm[self.spatial_key],
                adata.obs[self.label_key].to_list()
            )  # store data in cpp object

    def find_fp_knn(self,
                    ct: str,
                    dataset: Optional[Union[str, List[str]]] = None,
                    k: int = 30,
                    min_support: float = 0.5,
                    dis_duplicates: bool = False,
                    ):
        # Identify the datasets that match the given dataset name as well as containing the given cell type ct
        # intput indices of the matched datasets.
        dataset = self.check_dataset(dataset)

        # Get the id of FOVs of datasets
        valid_ds_names = [d.split('_')[0] for d in self.datasets]
        id_dataset = []
        for ds in dataset:
            id_dataset = id_dataset + [i for i, d in enumerate(valid_ds_names) if d == ds]

        ct = ct.replace('_', '-')
        if_exist_label = [ct in set(self.labels[i]) for i in id_dataset]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in specified datasets!")

        fp = self.spatial_query_multiple.find_fp_knn(
            cell_type=ct,
            fov_ids=id_dataset,
            k=k,
            min_support=min_support,
            dis_duplicates=dis_duplicates,
        )
        if len(fp) == 0:
            return pd.DataFrame(columns=['itemsets', 'support'])
        fp = pd.DataFrame(fp).sort_values(by='support', ascending=False, ignore_index=True)
        fp.rename(columns={'items': 'itemsets'}, inplace=True)
        return fp

    def find_fp_dist(self,
                     ct: str,
                     dataset: Optional[Union[str, List[str]]] = None,
                     max_dist: float = 100.0,
                     min_size: int = 0,
                     min_support: float = 0.5,
                     dis_duplicates: bool = False,
                     ):
        dataset = self.check_dataset(dataset)

        valid_ds_names = [d.split('_')[0] for d in self.datasets]
        id_dataset = []
        for ds in dataset:
            id_dataset = id_dataset + [i for i, d in enumerate(valid_ds_names) if d == ds]

        ct = ct.replace('_', '-')
        if_exist_label = [ct in set(self.labels[i]) for i in id_dataset]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in specified datasets!")

        fp = self.spatial_query_multiple.find_fp_dist(
            cell_type=ct,
            fov_ids=id_dataset,
            radius=max_dist,
            min_support=min_support,
            dis_duplicates=dis_duplicates,
            min_size=min_size
        )
        if len(fp) == 0:
            return pd.DataFrame(columns=['itemsets', 'support'])

        fp = pd.DataFrame(fp).sort_values(by='support', ascending=False, ignore_index=True)
        fp.rename(columns={'items': 'itemsets'}, inplace=True)
        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Optional[Union[str, List[str]]] = None,
                             dataset: Optional[Union[str, List[str]]] = None,
                             k: int = 30,
                             min_support: float = 0.5,
                             dis_duplicates: bool = False,
                             max_dist: float = 500,
                             ):
        dataset = self.check_dataset(dataset)

        valid_ds_names = [d.split('_')[0] for d in self.datasets]
        id_dataset = []
        for ds in dataset:
            id_dataset = id_dataset + [i for i, d in enumerate(valid_ds_names) if d == ds]

        ct = ct.replace('_', '-')
        if_exist_label = [ct in set(self.labels[i]) for i in id_dataset]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in specified datasets!")

        # Check whether specified motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        if motifs is None:
            start_time = time.time()
            fp = self.find_fp_knn(
                ct=ct,
                dataset=dataset,
                k=k,
                min_support=min_support,
                dis_duplicates=dis_duplicates,
            )
            motifs = fp['itemsets'].tolist()
            end_time = time.time()
            print(f"{end_time-start_time} seconds for identifying frequent patterns")
        else:
            # remove non-exist cell types in motifs
            start_time = time.time()
            if isinstance(motifs, str):
                motifs = [motifs]
            labels_valid = [list(set(self.labels[i])) for i in id_dataset]
            labels_valid = [l for labels_id in labels_valid for l in labels_id]
            labels_valid_unique = set(labels_valid)
            motifs_exc = [m for m in motifs if m not in labels_valid_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {dataset}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            if len(motifs) == 0:
                raise ValueError(f"All cell types in motifs are missed in {self.label_key}.")
            motifs = [
                motifs]  # make sure motifs is a List[List[str]] to be consistent with the outputs of frequent patterns
            end_time = time.time()
            print(f"{end_time-start_time} seconds for pre-processing motifs")

        start_time = time.time()
        out_enrichment = self.spatial_query_multiple.motif_enrichment_knn(
            cell_type=ct,
            motifs=motifs,
            fov_ids=id_dataset,
            k=k,
            min_support=min_support,
            dis_duplicates=dis_duplicates,
            max_dist=max_dist,
        )
        end_time = time.time()
        print(f"{end_time-start_time} seconds for cpp implementations")

        start_time = time.time()
        out = []
        for motif_count in out_enrichment:
            motif = sorted(motif_count['motifs'])
            hyge = hypergeom(M=motif_count['n_labels'],
                             n=motif_count['n_center'],
                             N=motif_count['n_motif'])
            motif_out = {'center': ct, 'motifs': motif, 'n_center_motif': motif_count['n_center_motif'],
                         'n_center': motif_count['n_center'], 'n_motif': motif_count['n_motif'],
                         'p-values': hyge.sf(motif_count['n_center_motif'])}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        end_time = time.time()
        print(f"{end_time - start_time} seconds for hypergeometric test")

        start_time = time.time()
        p_values = out_pd['p-values'].tolist()
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')
        out_pd['corrected p-values'] = corrected_p_values
        out_pd['if_significant'] = if_rejected
        out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
        end_time = time.time()
        print(f"{end_time-start_time} seconds for multi testing correction")
        return out_pd

    def motif_enrichment_dist(self,
                              ct: str,
                              motifs: Optional[Union[str, List[str]]] = None,
                              dataset: Optional[Union[str, List[str]]] = None,
                              max_dist: float = 100.0,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              dis_duplicates: bool = False,
                              ):
        dataset = self.check_dataset(dataset)

        valid_ds_names = [d.split('_')[0] for d in self.datasets]
        id_dataset = []
        for ds in dataset:
            id_dataset = id_dataset + [i for i, d in enumerate(valid_ds_names) if d == ds]

        ct = ct.replace('_', '-')
        if_exist_label = [ct in set(self.labels[i]) for i in id_dataset]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in specified datasets!")

        # Check whether specified motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        if motifs is None:
            fp = self.find_fp_dist(
                ct=ct,
                dataset=dataset,
                max_dist=max_dist,
                min_size=min_size,
                min_support=min_support,
                dis_duplicates=dis_duplicates,
            )
            motifs = fp['items'].tolist()
        else:
            # remove non-exist cell types in motifs
            start_time = time.time()
            if isinstance(motifs, str):
                motifs = [motifs]
            labels_valid = [list(set(self.labels[i])) for i in id_dataset]
            labels_valid = [l for labels_id in labels_valid for l in labels_id]
            labels_valid_unique = set(labels_valid)
            motifs_exc = [m for m in motifs if m not in labels_valid_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {dataset}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            if len(motifs) == 0:
                raise ValueError(f"All cell types in motifs are missed in {self.label_key}.")
            motifs = [
                motifs]  # make sure motifs is a List[List[str]] to be consistent with the outputs of frequent patterns
            end_time = time.time()
            print(f"{end_time-start_time} seconds for pre-processing motifs")

        start_time = time.time()
        out_enrichment = self.spatial_query_multiple.motif_enrichment_dist(
            cell_type=ct,
            motifs=motifs,
            fov_ids=id_dataset,
            radius=max_dist,
            min_support=min_support,
            dis_duplicates=dis_duplicates,
            min_size=min_size,
        )
        end_time = time.time()
        print(f"{end_time-start_time} seconds for cpp_implementations")

        start_time = time.time()
        out = []
        for motif_count in out_enrichment:
            motif = sorted(motif_count['motifs'])
            hyge = hypergeom(M=motif_count['n_labels'],
                             n=motif_count['n_center'],
                             N=motif_count['n_motif'])
            motif_out = {'center': ct, 'motifs': motif, 'n_center_motif': motif_count['n_center_motif'],
                         'n_center': motif_count['n_center'], 'n_motif': motif_count['n_motif'],
                         'p-values': hyge.sf(motif_count['n_center_motif'])}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        end_time = time.time()
        print(f"{end_time - start_time} seconds for hypergeometric test")

        start_time = time.time()
        p_values = out_pd['p-values'].tolist()
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')
        out_pd['corrected p-values'] = corrected_p_values
        out_pd['if_significant'] = if_rejected
        out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
        end_time = time.time()
        print(f"{end_time - start_time} seconds for multi testing correction")
        return out_pd

    def differential_analysis_knn(self,
                                  ct: str,
                                  datasets: List[str],
                                  k: int = 30,
                                  min_support: float = 0.5,
                                  ):
        # Use same codes but use replace the frequent patterns search with cpp methods
        datasets = self.check_dataset(datasets)
        if len(datasets) != 2:
            raise ValueError("Require 2 valid datasets for differential analysis.")

        valid_ds_names = [d.split('_')[0] for d in self.datasets]
        id_dataset0 = [i for i, d in enumerate(valid_ds_names) if d == datasets[0]]
        id_dataset1 = [i for i, d in enumerate(valid_ds_names) if d == datasets[1]]

        out = self.spatial_query_multiple.differential_analysis_knn(
            cell_type=ct,
            fovs_id0=id_dataset0,
            fovs_id1=id_dataset1,
            k=k,
            min_support=min_support
        )
        # Based on the format of out to see how to process later
        n_fovs0 = len(list(out[0].values())[0])
        n_fovs1 = len(list(out[1].values())[0])

        # Assign support values of each pattern in each fov.
        # Fov support is named using support_{dataset}_i.
        # Since there might be some FOVs without cell type ct,
        # hence 'i' is just used to index FOVs, instead of the real FOV ids of the pattern.
        fp0 = pd.DataFrame(out[0]).T
        fp0.columns = [f'support_{datasets[0]}_{i}' for i in range(n_fovs0)]
        fp1 = pd.DataFrame(out[1]).T
        fp1.columns = [f'support_{datasets[1]}_{i}' for i in range(n_fovs1)]
        fp_datasets = pd.merge(fp0, fp1, left_index=True, right_index=True, how='outer')
        fp_datasets.fillna(0, inplace=True)
        items_list = [s.split(", ") for s in fp_datasets.index]
        fp_datasets['itemsets'] = items_list

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
        fp_datasets['corrected p-values'] = corrected_p_values
        fp_datasets['if_significant'] = if_rejected
        fp_datasets['p-values'] = p_values

        # Return the significant patterns in each dataset
        fp_dataset0 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[0]) & (fp_datasets['if_significant'])
            ][['itemsets', 'p-values', 'corrected p-values']]
        fp_dataset1 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[1]) & (fp_datasets['if_significant'])
            ][['itemsets', 'p-values', 'corrected p-values']]
        fp_dataset0 = fp_dataset0.sort_values(by='corrected p-values', ascending=True)
        fp_dataset1 = fp_dataset1.sort_values(by='corrected p-values', ascending=True)
        return fp_dataset0, fp_dataset1

    def differential_analysis_dist(self,
                                   ct: str,
                                   datasets: List[str],
                                   max_dist: float = 100.0,
                                   min_support: float = 0.5,
                                   min_size: int = 0,
                                   ):
        # Use same codes but use replace the frequent patterns search with cpp methods
        datasets = self.check_dataset(datasets)
        if len(datasets) != 2:
            raise ValueError("Require 2 valid datasets for differential analysis.")

        valid_ds_names = [d.split('_')[0] for d in self.datasets]
        id_dataset0 = [i for i, d in enumerate(valid_ds_names) if d == datasets[0]]
        id_dataset1 = [i for i, d in enumerate(valid_ds_names) if d == datasets[1]]

        out = self.spatial_query_multiple.differential_analysis_dist(
            cell_type=ct,
            fovs_id0=id_dataset0,
            fovs_id1=id_dataset1,
            radius=max_dist,
            min_support=min_support,
            min_size=min_size
        )
        n_fovs0 = len(list(out[0].values())[0])
        n_fovs1 = len(list(out[1].values())[0])

        # Assign support values of each pattern in each fov.
        # Fov support is named using support_{dataset}_i.
        # Since there might be some FOVs without cell type ct,
        # hence 'i' is just used to index FOVs, instead of the real FOV ids of the pattern.
        fp0 = pd.DataFrame(out[0]).T
        fp0.columns = [f'support_{datasets[0]}_{i}' for i in range(n_fovs0)]
        fp1 = pd.DataFrame(out[1]).T
        fp1.columns = [f'support_{datasets[1]}_{i}' for i in range(n_fovs1)]
        fp_datasets = pd.merge(fp0, fp1, left_index=True, right_index=True, how='outer')
        fp_datasets.fillna(0, inplace=True)
        items_list = [s.split(", ") for s in fp_datasets.index]
        fp_datasets['itemsets'] = items_list

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
        fp_datasets['corrected p-values'] = corrected_p_values
        fp_datasets['if_significant'] = if_rejected
        fp_datasets['p-values'] = p_values

        # Return the significant patterns in each dataset
        fp_dataset0 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[0]) & (fp_datasets['if_significant'])
            ][['itemsets', 'p-values', 'corrected p-values']]
        fp_dataset1 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[1]) & (fp_datasets['if_significant'])
            ][['itemsets', 'p-values', 'corrected p-values']]
        fp_dataset0 = fp_dataset0.sort_values(by='corrected p-values', ascending=True)
        fp_dataset1 = fp_dataset1.sort_values(by='corrected p-values', ascending=True)
        return fp_dataset0, fp_dataset1

    def check_dataset(self,
                      dataset: Optional[Union[str, List[str]]] = None,
                      ):
        if dataset is None:
            # Use all datasets if dataset is not provided
            out = [d.split('_')[0] for d in self.datasets]
        else:
            if isinstance(dataset, str):
                dataset = [dataset]
            dataset = [d.replace("_", "-") for d in dataset]

            # test if the input dataset name is valid
            valid_ds_names = [d.split('_')[0] for d in self.datasets]
            out = []
            for ds in dataset:
                if ds not in valid_ds_names:
                    print(f"{ds} is not valid dataset names. Ignoring it.\n"
                          f"Please check valid names by obj.datasets")
                else:
                    out.append(ds)
            if len(out) == 0:
                raise ValueError("All input dataset names are invalid.")

        return list(set(out))
