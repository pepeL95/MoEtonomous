import umap
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from langchain_core.runnables import Runnable
from langchain_core.embeddings.embeddings import Embeddings
from transformers.models.mpnet.modeling_mpnet import MPNetModel
from transformers.models.mpnet.tokenization_mpnet_fast import MPNetTokenizerFast
from sklearn.mixture import GaussianMixture
from agents.base.agent import BaseAgent

RANDOM_SEED = 224  # Fixed seed for reproducibility


class RAPTOR:
    def __init__(
            self,
            agent: BaseAgent | Runnable,
            embd: Union[MPNetModel, Embeddings],
            tokenizer: MPNetTokenizerFast = None,
            device: str = None
    ) -> None:

        self.device = device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # Enable gpu (if ready)
        if self.device not in {'cuda', 'mps', 'cpu'}:
            raise ValueError(
                f"Device must be one of [cpu, cuda, mps]. Got {self.device}.")

        # Set embedding model and tokenizer(if needed)
        self.tokenizer = tokenizer
        self.embd = embd
        if isinstance(self.embd, MPNetModel):
            self.embd = embd.to(self.device)
            if self.tokenizer is None:
                raise ValueError(
                    "You must provide a tokenizer for the provided embedding model.")

        elif not isinstance(self.embd, Embeddings):
            raise ValueError(
                "The embedding model must be one of Union[MPNetModel, Embeddings]")

        # Set llm and summarization prompt
        self.agent = agent

    @staticmethod
    def _global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        '''
        Perform global dimensionality reduction on the embeddings using UMAP.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - n_neighbors: Optional; the number of neighbors to consider for each point.
                       If not provided, it defaults to the square root of the number of embeddings.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        '''
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    @staticmethod
    def _local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
    ) -> np.ndarray:
        '''
        Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - num_neighbors: The number of neighbors to consider for each point.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        '''
        return umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    @classmethod
    def _get_optimal_clusters(
        cls, embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
    ) -> int:
        '''
        Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - max_clusters: The maximum number of clusters to consider.
        - random_state: Seed for reproducibility.

        Returns:
        - An integer representing the optimal number of clusters found.
        '''
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    @classmethod
    def _GMM_cluster(cls, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        '''
        Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - threshold: The probability threshold for assigning an embedding to a cluster.
        - random_state: Seed for reproducibility.

        Returns:
        - A tuple containing the cluster labels and the number of clusters determined.
        '''
        n_clusters = cls._get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters,
                             random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters

    @classmethod
    def _perform_clustering(
        cls, embeddings: np.ndarray, dim: int, threshold: float
    ) -> List[np.ndarray]:
        '''
        Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
        using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for UMAP reduction.
        - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

        Returns:
        - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
        '''

        # Base case
        if len(embeddings) <= dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))]

        reduced_embeddings_global = cls._global_cluster_embeddings(
            embeddings, dim)
        global_clusters, n_global_clusters = cls._GMM_cluster(
            reduced_embeddings_global, threshold
        )

        # Keep grouping
        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        for i in range(n_global_clusters):
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                local_clusters = [np.array([0])
                                  for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                reduced_embeddings_local = cls._local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = cls._GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters

    def _embed(self, texts: List[str]) -> Union[torch.Tensor, np.ndarray]:
        '''    
        Parameters:
        - texts: List[str], a list of text documents to be embedded.

        Returns:
        - numpy.ndarray: An array of embeddings for the given text documents.
        '''
        embeddings = None
        if isinstance(self.embd, MPNetModel):
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.embd(**inputs)
            embeddings = outputs.last_hidden_state
        else:
            embeddings = self.embd.embed_documents(texts)
            embeddings = np.array(embeddings)
        return embeddings

    def _embed_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        '''
        Embeds texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

        This function combines embedding generation and clustering into a single step. It assumes the existence
        of a previously defined `perform_clustering` function that performs clustering on the embeddings.

        Parameters:
        - texts: List[str], a list of text documents to be processed.

        Returns:
        - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
        '''
        # Get embeddings as numpy array
        embeddings = self._embed(texts)
        if isinstance(embeddings, torch.Tensor):
            text_embeddings = torch.mean(embeddings, dim=1).to('cpu')
            text_embeddings_np = np.array(text_embeddings)
            # Cluster labels
            cluster_labels = self._perform_clustering(
                text_embeddings_np, 10, 0.1)

        # Otherwise, embeddings is already a numpy array
        else:
            # Cluster labels
            cluster_labels = self._perform_clustering(embeddings, 10, 0.1)

        df = pd.DataFrame()
        df["text"] = texts
        df["embd"] = list(embeddings.to('cpu')) if isinstance(
            embeddings, torch.Tensor) else list(embeddings)
        df["cluster"] = cluster_labels

        return df

    @staticmethod
    def _fmt_txt(df: pd.DataFrame) -> str:
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def _embed_cluster_summarize_texts(
        self, texts: List[str], level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        This method first generates embeddings for the texts,
        clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
        the content within each cluster.

        Parameters:
        - texts: A list of text documents to be processed.
        - level: An integer parameter that could define the depth or detail of processing.

        Returns:
        - Tuple containing two DataFrames:
          1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
          2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
         '''
        df_clusters = self._embed_cluster_texts(texts)
        expanded_list = []
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"],
                        "cluster": cluster}
                )
        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()

        print(f"--Generated {len(all_clusters)} clusters--")

        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self._fmt_txt(df_cluster)
            summaries.append(self.agent.invoke({"input": formatted_txt}))

        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )

        return df_clusters, df_summary

    def recursive_embed_cluster_summarize(
        self, texts: List[str], level: int = 1, n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        '''
        Parameters:
        - texts: List[str], texts to be processed.
        - level: int, current recursion level (starts at 1).
        - n_levels: int, maximum depth of recursion.

        Returns:
        - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
          levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
        '''
        results = {}

        df_clusters, df_summary = self._embed_cluster_summarize_texts(
            texts, level)
        results[level] = (df_clusters, df_summary)

        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )
            results.update(next_level_results)

        return results

    def flatten_tree(self, doc_tree: dict) -> Tuple[List[str], Union[torch.Tensor, np.ndarray]]:
        '''
        Returns the documents and summaries and their embeddings
        '''
        # all_texts = documents.copy()
        all_texts = []
        # Iterate through the results to extract summaries from each level and add them to all_texts
        for level in sorted(doc_tree.keys()):
            # Extract summaries from the current level's DataFrame
            summaries = doc_tree[level][1]["summaries"].tolist()

            # Extend all_texts with the summaries and texts from the current level
            all_texts = summaries + all_texts  # add summaries (actual)
            # add documents (actual)
            all_texts.extend(doc_tree[level][0]['text'].tolist())

        all_embeddings = self._embed(all_texts)

        return all_texts, all_embeddings
