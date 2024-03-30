from collections import Counter
from typing import Union, List

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.multitest as mt
from scipy.stats import hypergeom
from spatial_module import SpatialDataSingle


class spatial_query_single:
    def __init__(self,
                 adata: ad.AnnData,
                 dataset: str = 'ST',
                 spatial_key: str = 'X_spatial',
                 label_key: str = 'cell_type',
                 ):
        if spatial_key not in adata.obsm.keys() or label_key not in adata.obs.keys():
            raise ValueError(f"The Anndata object must contain {spatial_key} in obsm and {label_key} in obs.")
        # Store spatial position and cell type label
        self.spatial_key = spatial_key
        self.dataset = dataset
        self.label_key = label_key
        self.labels = adata.obs[self.label_key]
        self.labels = self.labels.astype('category')
        self.labels = [label.replace("_", "-") for label in
                       self.labels]  # use "_" as delimiter in the following analysis, so replace "_" with "-"
        self.spatial_query_single = SpatialDataSingle()
        self.spatial_query_single.set_data(adata.obsm[self.spatial_key], self.labels)  # store data in cpp codes
        self.spatial_query_single.build_kdtree()

    @staticmethod
    def has_motif(neighbors: List[str], labels: List[str]) -> bool:
        freq_neighbors = Counter(neighbors)
        freq_labels = Counter(labels)
        for element, count in freq_neighbors.items():
            if freq_labels[element] < count:
                return False

        return True

    def build_fptree_knn(self,
                         cell_pos: np.ndarray = None,
                         k: int = 30,
                         min_support: float = 0.5,
                         dis_duplicates: bool = False,
                         max_dist: float = 500,
                         if_max: bool = True
                         ):

        fp = self.spatial_query_single.build_fptree_knn(cell_pos=cell_pos, k=k, min_support=min_support,
                                                        dis_duplicates=dis_duplicates, if_max=if_max,
                                                        max_dist=max_dist)
        if len(fp) == 0:
            return pd.DataFrame(columns=['itemsets', 'support'])
        else:
            fp_df = pd.DataFrame(fp).sort_values(by='support', ignore_index=True, ascending=False)
            fp_df.rename(columns={'items': 'itemsets'}, inplace=True)
            return fp_df

    def build_fptree_dist(self,
                          cell_pos: np.ndarray = None,
                          dis_duplicates: bool = False,
                          max_dist: float = 100,
                          min_support: float = 0.5,
                          if_max: bool = True,
                          min_size: int = 0,
                          ):
        fp_tree, df, valid_idxs = self.spatial_query_single.build_fptree_dist(cell_pos=cell_pos,
                                                                              radius=max_dist,
                                                                              min_support=min_support,
                                                                              dis_duplicates=dis_duplicates,
                                                                              if_max=if_max,
                                                                              min_size=min_size,
                                                                              )
        if len(fp_tree) == 0:
            return pd.DataFrame(columns=['itemsets', 'support']), df, valid_idxs
        else:
            fp_df = pd.DataFrame(fp_tree).sort_values(by='support', ignore_index=True, ascending=False)
            fp_df.rename(columns={'items': 'itemsets'}, inplace=True)
            return fp_df, df, valid_idxs

    def find_fp_knn(self,
                    ct: str,
                    k: int = 30,
                    min_support: float = 0.5,
                    dis_duplicates: bool = False,
                    ):
        ct = ct.replace("_", "-")
        if ct not in set(self.labels):
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(self.labels) if l == ct]
        spatial_pos = self.spatial_query_single.get_coordinates()
        ct_pos = spatial_pos[cinds]
        fp = self.build_fptree_knn(cell_pos=ct_pos, k=k,
                                   min_support=min_support,
                                   dis_duplicates=dis_duplicates)
        return fp

    def find_fp_dist(self,
                     ct: str,
                     dis_duplicates: bool = False,
                     max_dist: float = 100,
                     min_support: float = 0.5,
                     min_size: int = 0,
                     ):
        ct = ct.replace("_", "-")
        if ct not in set(self.labels):
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(self.labels) if l == ct]
        spatial_pos = self.spatial_query_single.get_coordinates()
        ct_pos = spatial_pos[cinds]

        fp, _, _ = self.build_fptree_dist(cell_pos=ct_pos,
                                          dis_duplicates=dis_duplicates,
                                          max_dist=max_dist,
                                          min_support=min_support,
                                          min_size=min_size,
                                          )
        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Union[str, List[str], List[List[str]]] = None,
                             k: int = 30,
                             min_support: float = 0.5,
                             dis_duplicates: bool = False,
                             max_dist: float = 100,
                             ):
        ct = ct.replace("_", "-")
        if ct not in set(self.labels):
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        # Ensure the input of motifs is a list of lists of strings, i.e., List[List[str]].
        if motifs is None:
            fp = self.find_fp_knn(ct=ct, k=k, min_support=min_support, dis_duplicates=dis_duplicates)
            motifs = fp['itemsets']
            motifs = [m for m in motifs]
        elif isinstance(motifs, str):
            motifs = [[motifs]]
        elif isinstance(motifs, list) and all(isinstance(item, str) for item in motifs):
            motifs = [motifs]
        elif isinstance(motifs, list) and all(isinstance(item, list) for item in motifs):
            motifs = motifs

        # Exclude non-existing cell types in motifs.
        labels_unique = set(self.labels)
        motifs_exc = [m for motif in motifs for m in motif if m not in labels_unique]
        if len(motifs_exc) > 0:
            print(f"Found no {motifs_exc} in {self.label_key}. Ignoring them.")

            motifs = [[m for m in motif if m not in motifs_exc] for motif in motifs]

        if len(motifs) == 0:
            raise ValueError("No frequent patterns were found. Please lower min_support value.")

        out = self.spatial_query_single.motif_enrichment_knn(ct=ct,
                                                             motifs=motifs,
                                                             k=k,
                                                             min_support=min_support,
                                                             dis_duplicates=dis_duplicates,
                                                             max_dist=max_dist)
        # Perform hyper-geometric tests based on out values.
        for motif_out_dict in out:
            motif = motif_out_dict['motifs']
            n_ct = motif_out_dict['n_center']
            n_motif_labels = motif_out_dict['n_motif']
            n_center_motif = motif_out_dict['n_center_motif']
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))
            hyge = hypergeom(M=len(self.labels), n=n_ct, N=n_motif_labels)
            p_val = hyge.sf(n_center_motif)
            motif_out_dict['p-values'] = p_val

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
                              motifs: Union[str, List[str]] = None,
                              dis_duplicates: bool = False,
                              max_dist: float = 100,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              ):
        ct = ct.replace("_", "-")
        if ct not in set(self.labels):
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        # Ensure the input of motifs is a list of lists of strings, i.e., List[List[str]].
        if motifs is None:
            fp = self.find_fp_dist(ct=ct, max_dist=max_dist, min_support=min_support, dis_duplicates=dis_duplicates,
                                   min_size=min_size)
            motifs = fp['itemsets']
            motifs = [m for m in motifs]
        elif isinstance(motifs, str):
            motifs = [[motifs]]
        elif isinstance(motifs, list) and all(isinstance(item, str) for item in motifs):
            motifs = [motifs]
        elif isinstance(motifs, list) and all(isinstance(item, list) for item in motifs):
            motifs = motifs

        # Exclude non-existing cell types in motifs.
        labels_unique = set(self.labels)
        motifs_exc = [m for motif in motifs for m in motif if m not in labels_unique]
        if len(motifs_exc) > 0:
            print(f"Found no {motifs_exc} in {self.label_key}. Ignoring them.")

            motifs = [[m for m in motif if m not in motifs_exc] for motif in motifs]

        if len(motifs) == 0:
            raise ValueError("No frequent patterns were found. Please lower min_support value.")

        out = self.spatial_query_single.motif_enrichment_dist(ct=ct,
                                                              motifs=motifs,
                                                              radius=max_dist,
                                                              min_support=min_support,
                                                              dis_duplicates=dis_duplicates,
                                                              min_size=min_size)
        # Perform hypergeom tests based on out values.
        for motif_out_dict in out:
            motif = motif_out_dict['motifs']
            n_ct = motif_out_dict['n_center']
            n_motif_labels = motif_out_dict['n_motif']
            n_center_motif = motif_out_dict['n_center_motif']
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))
            hyge = hypergeom(M=len(self.labels), n=n_ct, N=n_motif_labels)
            p_val = hyge.sf(n_center_motif)
            motif_out_dict['p-values'] = p_val

        out_pd = pd.DataFrame(out)
        p_values = out_pd['p-values'].tolist()
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')
        out_pd['corrected p-values'] = corrected_p_values
        out_pd['if_significant'] = if_rejected
        out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
        return out_pd

    def find_patterns_grid(self,
                           max_dist: float = 100,
                           min_size: int = 0,
                           min_support: float = 0.5,
                           dis_duplicates: bool = False,
                           if_display: bool = True,
                           fig_size: tuple = (10, 5),
                           return_cellID: bool = False,
                           return_grid: bool = False,
                           ):
        spatial_pos = self.spatial_query_single.get_coordinates()
        xmax, ymax = np.max(spatial_pos, axis=0)
        xmin, ymin = np.min(spatial_pos, axis=0)
        x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
        y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
        grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

        fp, trans_df, idxs = self.build_fptree_dist(cell_pos=grid,
                                                    dis_duplicates=dis_duplicates,
                                                    max_dist=max_dist,
                                                    if_max=True,
                                                    min_size=min_size,
                                                    min_support=min_support)
        id_neighbor_motifs = []
        if if_display or return_cellID:
            for motif in fp['itemsets']:
                motif = list(motif)
                fp_spots_index = set()
                ct_counts_in_motif = pd.Series(motif).value_counts().to_dict()
                ids = [i for i, d in enumerate(trans_df) if
                       all(d.get(k, 0) >= v for k, v in ct_counts_in_motif.items())]
                if len(ids) > 0:
                    # ids = ids.index[ids == True].to_list()
                    fp_spots_index.update([i for id in ids for i in idxs[id] if self.labels[i] in motif])
                id_neighbor_motifs.append(fp_spots_index)
        if return_cellID:
            fp['cell_id'] = id_neighbor_motifs

        if if_display:
            fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
            n_colors = len(fp_cts)
            colors = sns.color_palette('hsv', n_colors)
            color_map = {ct: col for ct, col in zip(fp_cts, colors)}

            fp_spots_index = set()
            for cell_id in id_neighbor_motifs:
                fp_spots_index.update(cell_id)

            fp_spot_pos = spatial_pos[list(fp_spots_index), :]
            fp_spot_label = [self.labels[i] for i in fp_spots_index]
            fig, ax = plt.subplots(figsize=fig_size)
            # Plotting the grid lines
            for x in x_grid:
                ax.axvline(x, color='lightgray', linestyle='--', lw=0.5)

            for y in y_grid:
                ax.axhline(y, color='lightgray', linestyle='--', lw=0.5)

            for ct in fp_cts:
                ct_ind = [i for i, c in enumerate(fp_spot_label) if c == ct]
                ax.scatter(fp_spot_pos[ct_ind, 0], fp_spot_pos[ct_ind, 1],
                           label=ct, color=color_map[ct], s=1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
            plt.xlabel('Spatial X')
            plt.ylabel('Spatial Y')
            plt.title('Spatial distribution of frequent patterns')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout(rect=[0, 0, 1.1, 1])
            plt.show()

        if return_grid:
            return fp.sort_values(by='support', ignore_index=True, ascending=False), grid
        else:
            return fp.sort_values(by='support', ignore_index=True, ascending=False)

    def find_patterns_rand(self,
                           max_dist: float = 100,
                           n_points: int = 1000,
                           min_support: float = 0.5,
                           dis_duplicates: bool = False,
                           min_size: int = 0,
                           if_display: bool = True,
                           fig_size: tuple = (10, 5),
                           return_cellID: bool = False,
                           seed: int = 2023):
        spatial_pos = self.spatial_query_single.get_coordinates()
        xmax, ymax = np.max(spatial_pos, axis=0)
        xmin, ymin = np.min(spatial_pos, axis=0)
        np.random.seed(seed)
        pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                               np.random.rand(n_points) * (ymax - ymin) + ymin))

        fp, trans_df, idxs = self.build_fptree_dist(cell_pos=pos,
                                                    dis_duplicates=dis_duplicates,
                                                    max_dist=max_dist, min_size=min_size,
                                                    min_support=min_support,
                                                    )
        id_neighbor_motifs = []
        if if_display or return_cellID:
            for motif in fp['itemsets']:
                motif = list(motif)
                fp_spots_index = set()
                ct_counts_in_motif = pd.Series(motif).value_counts().to_dict()
                ids = [i for i, d in enumerate(trans_df) if
                       all(d.get(k, 0) >= v for k, v in ct_counts_in_motif.items())]
                if len(ids) > 0:
                    fp_spots_index.update([i for id in ids for i in idxs[id] if self.labels[i] in motif])
                id_neighbor_motifs.append(fp_spots_index)
        if return_cellID:
            fp['cell_id'] = id_neighbor_motifs

        if if_display:
            fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
            n_colors = len(fp_cts)
            colors = sns.color_palette('hsv', n_colors)
            color_map = {ct: col for ct, col in zip(fp_cts, colors)}

            fp_spots_index = set()
            for cell_id in id_neighbor_motifs:
                fp_spots_index.update(cell_id)

            fp_spot_pos = spatial_pos[list(fp_spots_index), :]
            fp_spot_label = [self.labels[i] for i in fp_spots_index]
            fig, ax = plt.subplots(figsize=fig_size)
            for ct in fp_cts:
                ct_ind = [i for i, c in enumerate(fp_spot_label) if c == ct]
                ax.scatter(fp_spot_pos[ct_ind, 0], fp_spot_pos[ct_ind, 1],
                           label=ct, color=color_map[ct], s=1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
            plt.xlabel('Spatial X')
            plt.ylabel('Spatial Y')
            plt.title('Spatial distribution of frequent patterns')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout(rect=[0, 0, 1.1, 1])
            plt.show()

        return fp.sort_values(by='support', ignore_index=True, ascending=False)

    def plot_fov(self,
                 min_cells_label: int = 50,
                 title: str = 'Spatial distribution of cell types',
                 fig_size: tuple = (10, 5)):

        spatial_pos = self.spatial_query_single.get_coordinates()
        cell_type_counts = Counter(self.labels)
        n_colors = sum(count >= min_cells_label for count in cell_type_counts.values())
        colors = sns.color_palette('hsv', n_colors)

        color_counter = 0
        fig, ax = plt.subplots(figsize=fig_size)

        # Iterate over each cell type
        for cell_type in sorted(set(self.labels)):
            # Filter data for each cell type
            index = [i for i, l in enumerate(self.labels) if l == cell_type]
            # Check if the cell type count is above the threshold
            if cell_type_counts[cell_type] >= min_cells_label:
                ax.scatter(spatial_pos[index, 0], spatial_pos[index, 1],
                           label=cell_type, color=colors[color_counter], s=1)
                color_counter += 1
            else:
                ax.scatter(spatial_pos[index, 0], spatial_pos[index, 1],
                           color='grey', s=1)

        handles, labels = ax.get_legend_handles_labels()

        # Modify labels to include count values
        new_labels = [f'{label} ({cell_type_counts[label]})' for label in labels]

        # Create new legend
        ax.legend(handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        # ax.legend(handles, new_labels, loc='lower center', bbox_to_anchor=(1, 0.5), markerscale=4)

        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)

        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title(title)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        # Adjust layout to prevent clipping of ylabel and accommodate the legend
        plt.tight_layout(rect=[0, 0, 1.1, 1])

        plt.show()

    def plot_motif_grid(self,
                        motif: Union[str, List[str]],
                        fp: pd.DataFrame,
                        fig_size: tuple = (10, 5),
                        max_dist: float = 100,
                        ):

        if isinstance(motif, str):
            motif = [motif]

        spatial_pos = self.spatial_query_single.get_coordinates()
        labels_unique = set(self.labels)
        motif_exc = [m for m in motif if m not in labels_unique]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        # Build mesh
        xmax, ymax = np.max(spatial_pos, axis=0)
        xmin, ymin = np.min(spatial_pos, axis=0)
        x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
        y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
        grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

        # self.build_fptree_dist returns valid_idxs () instead of all the idxs,
        # so recalculate the idxs directly using self.kd_tree.query_ball_point
        idxs = self.spatial_query_single.radius_search(cell_pos=grid, radius=max_dist)

        # Locate the index of grid points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [self.labels[id] for id in idx[1:]]
            if self.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above grid points with motif nearby
        id_motif_celltype = fp[fp['itemsets'].apply(
            lambda p: set(p)) == set(motif)]
        id_motif_celltype = id_motif_celltype['cell_id'].iloc[0]

        # Plot above spots and center grid points
        # Set color map as in find_patterns_grid
        fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
        n_colors = len(fp_cts)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(fp_cts, colors)}

        motif_spot_pos = spatial_pos[list(id_motif_celltype), :]
        motif_spot_label = [self.labels[i] for i in id_motif_celltype]
        fig, ax = plt.subplots(figsize=fig_size)
        # Plotting the grid lines
        for x in x_grid:
            ax.axvline(x, color='lightgray', linestyle='--', lw=0.5)

        for y in y_grid:
            ax.axhline(y, color='lightgray', linestyle='--', lw=0.5)
        ax.scatter(grid[id_center, 0], grid[id_center, 1], label='Grid Points',
                   edgecolors='red', facecolors='none', s=8)

        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(self.labels) if
                    i not in id_motif_celltype]  # the other spots are colored as background
        # bg_adata = self.adata[bg_index, :]
        bg_pos = spatial_pos[bg_index, :]
        ax.scatter(bg_pos[:, 0],
                   bg_pos[:, 1],
                   color='darkgrey', s=1)

        motif_unique = list(set(motif))
        for ct in motif_unique:
            ct_ind = [i for i, l in enumerate(motif_spot_label) if l == ct]
            ax.scatter(motif_spot_pos[ct_ind, 0],
                       motif_spot_pos[ct_ind, 1],
                       label=ct, color=color_map[ct], s=1)

        ax.set_xlim([xmin - max_dist, xmax + max_dist])
        ax.set_ylim([ymin - max_dist, ymax + max_dist])
        ax.legend(title='motif', loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        # ax.legend(title='motif', loc='lower center', bbox_to_anchor=(0, 0.), markerscale=4)
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title('Spatial distribution of frequent patterns')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1.1, 1])
        plt.show()

    def plot_motif_rand(self,
                        motif: Union[str, List[str]],
                        fp: pd.DataFrame,
                        max_dist: float = 100,
                        n_points: int = 1000,
                        fig_size: tuple = (10, 5),
                        seed: int = 2023,
                        ):
        if isinstance(motif, str):
            motif = [motif]

        spatial_pos = self.spatial_query_single.get_coordinates()
        labels_unique = set(self.labels)
        motif_exc = [m for m in motif if m not in labels_unique]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        # Random sample points
        xmax, ymax = np.max(spatial_pos, axis=0)
        xmin, ymin = np.min(spatial_pos, axis=0)
        np.random.seed(seed)
        pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                               np.random.rand(n_points) * (ymax - ymin) + ymin))

        idxs = self.spatial_query_single.radius_search(cell_pos=pos, radius=max_dist)

        # Locate the index of grid points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [self.labels[id] for id in idx[1:]]
            if self.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above random points with motif nearby
        id_motif_celltype = fp[fp['itemsets'].apply(
            lambda p: set(p)) == set(motif)]
        id_motif_celltype = id_motif_celltype['cell_id'].iloc[0]

        # Plot above spots and center grid points
        # Set color map as in find_patterns_grid
        fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
        n_colors = len(fp_cts)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(fp_cts, colors)}

        motif_spot_pos = spatial_pos[list(id_motif_celltype), :]
        motif_spot_label = [self.labels[i] for i in id_motif_celltype]
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(pos[id_center, 0], pos[id_center, 1], label='Random Sampling Points',
                   edgecolors='red', facecolors='none', s=8)

        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(self.labels) if
                    i not in id_motif_celltype]  # the other spots are colored as background
        bg_adata = spatial_pos[bg_index, :]
        ax.scatter(bg_adata[:, 0],
                   bg_adata[:, 1],
                   color='darkgrey', s=1)
        motif_unique = list(set(motif))
        for ct in motif_unique:
            ct_ind = [i for i, l in enumerate(motif_spot_label) if l == ct]
            ax.scatter(motif_spot_pos[ct_ind, 0],
                       motif_spot_pos[ct_ind, 1],
                       label=ct, color=color_map[ct], s=1)

        ax.set_xlim([xmin - max_dist, xmax + max_dist])
        ax.set_ylim([ymin - max_dist, ymax + max_dist])
        ax.legend(title='motif', loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title('Spatial distribution of frequent patterns')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1.1, 1])
        plt.show()

    def plot_motif_celltype(self,
                            ct: str,
                            motif: Union[str, List[str]],
                            max_dist: float = 100,
                            fig_size: tuple = (10, 5)
                            ):
        spatial_pos = self.spatial_query_single.get_coordinates()
        if isinstance(motif, str):
            motif = [motif]

        motif_exc = [m for m in motif if m not in set(self.labels)]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        if ct not in set(self.labels):
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [i for i, label in enumerate(self.labels) if label == ct]  # id of center cell type
        # ct_pos = self.spatial_pos[cinds]
        idxs = self.spatial_query_single.radius_search(cell_pos=spatial_pos, radius=max_dist)

        # find the index of cell type spots whose neighborhoods contain given motif
        cind_with_motif = []
        sort_motif = sorted(motif)
        for id in cinds:
            if self.has_motif(sort_motif, [self.labels[idx] for idx in idxs[id][1:]]):
                cind_with_motif.append(id)

        # Locate the index of motifs in the neighborhood of center cell type.
        id_motif_celltype = set()
        for id in cind_with_motif:
            id_neighbor = [i for i in idxs[id][1:] if self.labels[i] in motif]
            id_motif_celltype.update(id_neighbor)

        # Plot figures
        motif_unique = set(motif)
        n_colors = len(motif_unique)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(motif_unique, colors)}
        motif_spot_pos = spatial_pos[list(id_motif_celltype), :]
        motif_spot_label = [self.labels[i] for i in id_motif_celltype]
        fig, ax = plt.subplots(figsize=fig_size)
        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(self.labels) if i not in list(id_motif_celltype) + cind_with_motif]
        bg_adata = spatial_pos[bg_index, :]
        ax.scatter(bg_adata[:, 0],
                   bg_adata[:, 1],
                   color='darkgrey', s=1)
        # Plot center the cell type whose neighborhood contains motif
        ax.scatter(spatial_pos[cind_with_motif, 0],
                   spatial_pos[cind_with_motif, 1],
                   label=ct, edgecolors='red', facecolors='none', s=3,
                   )
        for ct_m in motif_unique:
            ct_ind = [i for i, l in enumerate(motif_spot_label) if l == ct_m]
            ax.scatter(motif_spot_pos[ct_ind, 0],
                       motif_spot_pos[ct_ind, 1],
                       label=ct_m, color=color_map[ct_m], s=1)

        ax.legend(title='motif', loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        # ax.legend(title='motif', loc='lower center', bbox_to_anchor=(1, 0.5), markerscale=4)
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title(f"Spatial distribution of motif around {ct}")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1.1, 1])
        plt.show()
