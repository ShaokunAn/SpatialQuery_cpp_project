from typing import List, Optional, Union

import pandas as pd
import statsmodels.stats.multitest as mt
from anndata import AnnData
from scipy.stats import hypergeom
from spatial_module import SpatialDataMultiple


class spatial_query_multi:
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
                adata.obs[self.label_key]
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
        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Optional = None,
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
            fp = self.find_fp_knn(
                ct=ct,
                dataset=dataset,
                k=k,
                min_support=min_support,
                dis_duplicates=dis_duplicates,
            )
            motifs = fp['items']
        else:
            # remove non-exist cell types in motifs
            if isinstance(motifs, str):
                motifs = [motifs]
            labels_valid = [self.labels[i] for i in id_dataset]
            labels_valid_unique = set(labels_valid)
            motifs_exc = [m for m in motifs if m not in labels_valid_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {dataset}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            if len(motifs) == 0:
                raise ValueError(f"All cell types in motifs are missed in {self.label_key}.")
            motifs = [
                motifs]  # make sure motifs is a List[List[str]] to be consistent with the outputs of frequent patterns

        out_enrichment = self.spatial_query_multiple.motif_enrichment_knn(
            cell_type=ct,
            motifs=motifs,
            fov_ids=id_dataset,
            k=k,
            min_support=min_support,
            dis_duplicates=dis_duplicates,
            max_dist=max_dist,
        )

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

        p_values = out_pd['p-values'].tolist()
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')
        out_pd['corrected p-values'] = corrected_p_values
        out_pd['if_significant'] = if_rejected
        out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
        return out_pd

    def motif_enrichment_dist(self,
                              ct: str,
                              motifs: Optional = None,
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
            motifs = fp['items']
        else:
            # remove non-exist cell types in motifs
            if isinstance(motifs, str):
                motifs = [motifs]
            labels_valid = [self.labels[i] for i in id_dataset]
            labels_valid_unique = set(labels_valid)
            motifs_exc = [m for m in motifs if m not in labels_valid_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {dataset}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            if len(motifs) == 0:
                raise ValueError(f"All cell types in motifs are missed in {self.label_key}.")
            motifs = [
                motifs]  # make sure motifs is a List[List[str]] to be consistent with the outputs of frequent patterns

        out_enrichment = self.spatial_query_multiple.motif_enrichment_dist(
            cell_type=ct,
            motifs=motifs,
            fov_ids=id_dataset,
            radius=max_dist,
            min_support=min_support,
            dis_duplicates=dis_duplicates,
            min_size=min_size,
        )

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

        p_values = out_pd['p-values'].tolist()
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')
        out_pd['corrected p-values'] = corrected_p_values
        out_pd['if_significant'] = if_rejected
        out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
        return out_pd

    def differential_analysis_dist(self,
                                   ct: str,
                                   datasets: Optional[Union[str, List[str]]],
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
            datasets=datasets,
            fovs_id0=id_dataset0,
            fovs_id1=id_dataset1,
            radius=max_dist,
            min_support=min_support,
            min_size=min_size
        )
        # Based on the format of out to see how to process later
        print('Done!')
        return out

    def differential_analysis_knn(self,
                                  ct: str,
                                  datasets: Optional[Union[str, List[str]]],
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
            datasets=datasets,
            fovs_id0=id_dataset0,
            fovs_id1=id_dataset1,
            k=k,
            min_support=min_support
        )
        # Based on the format of out to see how to process later
        print('Done!')
        return out

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

        return out
