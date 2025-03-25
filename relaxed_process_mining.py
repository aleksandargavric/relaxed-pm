#!/usr/bin/env python3

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random

########################################################################
#               1. Read and Prepare an Event Log with PM4Py            #
########################################################################

def read_event_log(xes_path: str):
    """
    Reads an XES event log file using PM4Py and returns the log object.
    """
    log = xes_importer.apply(xes_path)
    print(f"Successfully read log from: {xes_path}")
    return log


def extract_event_descriptions(log):
    """
    Extracts descriptive text from each event. For demonstration, we
    combine certain attributes (activity, resource) into a single string.
    
    Returns:
        A list of (case_id, event_idx, description_string).
    """
    descriptions = []
    for trace_idx, trace in enumerate(log):
        for event_idx, event in enumerate(trace):
            # Extract some textual attributes. Adjust as needed for your log structure.
            activity = event["concept:name"] if "concept:name" in event else "N/A"
            resource = event.get("org:resource", "UnknownResource")
            desc_str = f"Activity: {activity}; Resource: {resource}"
            descriptions.append((trace_idx, event_idx, desc_str))
    return descriptions


########################################################################
#      2. Embedding Generation (Text-Only or Multimodal)               #
########################################################################

def get_text_embeddings(events):
    """
    Placeholder function for obtaining text embeddings from, e.g., Ollama.
    Replace with an actual embedding call (e.g., using the 'nomic-embed-text' model).
    
    Args:
        events: List of event description strings.
    Returns:
        A NumPy array of shape (num_events, embedding_dim).
    """
    # For demonstration, return random vectors to mimic embeddings.
    # Replace this with your actual embedding calls, e.g.,
    #     embeddings = []
    #     for e in events:
    #         emb = get_embeddings_from_ollama(e, model='nomic-embed-text')
    #         embeddings.append(emb)
    #     return np.array(embeddings)
    num_events = len(events)
    embedding_dim = 8  # e.g., 768 or 1024 depending on real model
    return np.random.randn(num_events, embedding_dim)

def get_multimodal_embeddings(events):
    """
    Placeholder function for obtaining multimodal embeddings from, e.g., ImageBind.
    This could handle images, audio, text, etc., and unify them.
    
    Args:
        events: List of event description (or references to multimodal content).
    Returns:
        A NumPy array of shape (num_events, embedding_dim).
    """
    # Replace with your real calls to ImageBind or other libraries.
    num_events = len(events)
    embedding_dim = 8
    return np.random.randn(num_events, embedding_dim)


def compute_event_embeddings(descriptions, is_multimodal=False):
    """
    High-level function that decides whether to call text-only or multimodal embedding logic.
    """
    # `descriptions` is a list of (case_id, event_idx, description_string).
    # We only need the textual description here, but in a multimodal pipeline
    # you might retrieve references to image/audio frames, etc.
    event_texts = [d[-1] for d in descriptions]

    if is_multimodal:
        print("Computing embeddings in multimodal mode (placeholder).")
        embeddings = get_multimodal_embeddings(event_texts)
    else:
        print("Computing embeddings in text-only mode (placeholder).")
        embeddings = get_text_embeddings(event_texts)

    return embeddings


########################################################################
#      3. Cosine Similarity and Clustering (Hierarchical Example)       #
########################################################################

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two vectors a and b.
    
    Args:
        a, b: 1D NumPy arrays.
    Returns:
        Cosine similarity (float), range [-1, 1].
    """
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b + 1e-12)


def cluster_embeddings(embeddings, n_clusters=2):
    """
    Clusters embeddings using hierarchical (agglomerative) clustering.
    
    Args:
        embeddings: A NumPy array of shape (num_events, embedding_dim).
        n_clusters: Number of top-level clusters to produce.
    Returns:
        cluster_labels: An array of the same length as embeddings,
                        with an integer cluster assignment for each event.
    """
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering_model.fit_predict(embeddings)
    return cluster_labels


def visualize_dendrogram(embeddings, labels):
    """
    Creates a dendrogram from the hierarchical clustering results using Plotly.
    This is optional but can be helpful to illustrate cluster structures.
    """
    fig = ff.create_dendrogram(embeddings, labels=labels, orientation='right')
    fig.update_layout(width=800, height=800)
    fig.show()


def visualize_tsne(embeddings, cluster_labels):
    """
    Reduces embedding dimensions to 2D with t-SNE and shows a scatter plot
    colored by cluster labels.
    """
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embedding_2d = tsne.fit_transform(embeddings)
    
    # Convert labels to str to color by cluster
    cluster_labels_str = [f"Cluster_{lbl}" for lbl in cluster_labels]
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                # You can also vary color by cluster, but no subplots or style changes as requested:
                # color=cluster_labels,
            ),
            text=cluster_labels_str,
            hoverinfo='text'
        )
    )
    fig.update_layout(
        title="t-SNE Visualization of Event Embeddings",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2"
    )
    fig.show()


########################################################################
#      4. LLM-Based Labeling (Placeholder)                              #
########################################################################

def call_llm_to_label_cluster(events_in_cluster):
    """
    Placeholder: In your real flow, call an LLM like `llama3.3` or others
    to generate a human-readable cluster name based on the event texts.
    
    Args:
        events_in_cluster: A list of event descriptions in the same cluster.
    Returns:
        A string cluster name suggested by the LLM.
    """
    # Replace with real LLM call:
    # cluster_name = my_llm_api(events_in_cluster)
    # return cluster_name
    # For demonstration, just return a mock name.
    return "ClusterLabel_" + str(random.randint(100, 999))


def assign_cluster_labels_to_events(descriptions, cluster_assignments):
    """
    Groups events by their cluster labels, calls an LLM (placeholder) to get
    a descriptive name, and assigns the new name to each event.
    
    Args:
        descriptions: A list of (case_id, event_idx, description_string).
        cluster_assignments: 1D array with cluster indices for each event.
    Returns:
        A dictionary mapping (case_id, event_idx) -> new_label.
    """
    # Group events by cluster
    from collections import defaultdict
    cluster_groups = defaultdict(list)
    for (c_id, e_idx, desc), cluster_id in zip(descriptions, cluster_assignments):
        cluster_groups[cluster_id].append(desc)
    
    # For each cluster, call an LLM to get a name
    cluster_id_to_name = {}
    for cluster_id, event_texts in cluster_groups.items():
        cluster_label = call_llm_to_label_cluster(event_texts)
        cluster_id_to_name[cluster_id] = cluster_label

    # Build a mapping for all events
    event_to_cluster_label = {}
    for i, (c_id, e_idx, _) in enumerate(descriptions):
        event_to_cluster_label[(c_id, e_idx)] = cluster_id_to_name[cluster_assignments[i]]

    return event_to_cluster_label


########################################################################
#      5. Augment the Event Log with Cluster Labels for PM4Py          #
########################################################################

def augment_event_log_with_cluster_names(log, event_label_map):
    """
    Takes a PM4Py event log and a map of (case_id, event_idx) -> cluster_label,
    and inserts that label into the event attributes (e.g., concept:name).
    
    Returns:
        A modified copy of the log with an additional attribute, e.g. "cluster:label".
    """
    for trace_idx, trace in enumerate(log):
        for event_idx, event in enumerate(trace):
            new_label = event_label_map.get((trace_idx, event_idx), "NoClusterLabel")
            event["cluster:label"] = new_label
    return log


########################################################################
#      6. Run Process Discovery and Conformance Checking               #
########################################################################

def discover_and_view_process(log, discovery_algorithm="inductive"):
    """
    Uses PM4Py to discover a Petri net from the event log and visualize it.
    """
    if discovery_algorithm.lower() == "inductive":
        net, im, fm = pm4py.discover_petri_net_inductive(log)
    elif discovery_algorithm.lower() == "alpha":
        net, im, fm = pm4py.discover_petri_net_alpha(log)
    elif discovery_algorithm.lower() == "heuristic":
        net, im, fm = pm4py.discover_petri_net_heuristics(log)
    else:
        print(f"Unknown discovery algorithm '{discovery_algorithm}'. Defaulting to Inductive Miner.")
        net, im, fm = pm4py.discover_petri_net_inductive(log)
        
    print(f"Discovered Petri net with: {discovery_algorithm}")
    pm4py.view_petri_net(net, im, fm, format="svg")
    return net, im, fm


def check_conformance(log, net, im, fm):
    """
    Runs token-based replay to evaluate conformance between the Petri net
    and the event log.
    """
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    result = token_replay.apply(log, net, im, fm)
    # result is a list of dicts, one per trace, containing performance metrics
    # e.g., 'trace_fitness'
    fitness_values = [trace_data["trace_fitness"] for trace_data in result]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    print(f"Average trace fitness: {avg_fitness:.4f}")
    return avg_fitness


########################################################################
#      7. End-to-End Main Function                                     #
########################################################################

def main():
    # --- Step 1: Read event log ---
    # Replace with an actual path to your .xes file
    xes_path = "pm4py/tests/input_data/receipt.xes"
    log = read_event_log(xes_path)

    # --- Step 2: Extract text from events ---
    descriptions = extract_event_descriptions(log)
    print(f"Number of events extracted: {len(descriptions)}")

    # --- Step 3: Compute embeddings (choose text vs. multimodal) ---
    # is_multimodal = True/False depending on your use case
    is_multimodal = False
    embeddings = compute_event_embeddings(descriptions, is_multimodal=is_multimodal)

    # --- (Optional) Validate embedding similarity ---
    # For demonstration, let's compare the first two embeddings
    if len(embeddings) >= 2:
        sim_score = cosine_similarity(embeddings[0], embeddings[1])
        print(f"Cosine similarity between first two embeddings: {sim_score:.4f}")

    # --- Step 4: Cluster the embeddings ---
    cluster_assignments = cluster_embeddings(embeddings, n_clusters=2)
    print("Cluster assignments:", cluster_assignments[:10], "...")
    
    # --- (Optional) Visualize cluster structure ---
    visualize_dendrogram(embeddings, labels=None)   # You can pass in text labels if desired
    visualize_tsne(embeddings, cluster_assignments)
    
    # --- Step 5: Use LLM to name each cluster ---
    event_label_map = assign_cluster_labels_to_events(descriptions, cluster_assignments)

    # --- Step 6: Augment the log with new cluster labels ---
    augmented_log = augment_event_log_with_cluster_names(log, event_label_map)

    # --- Step 7: Discover a Petri net model from augmented log ---
    net, im, fm = discover_and_view_process(augmented_log, discovery_algorithm="inductive")

    # --- Perform conformance checking ---
    fitness = check_conformance(augmented_log, net, im, fm)
    print(f"Final conformance fitness on augmented log: {fitness:.4f}")


if __name__ == "__main__":
    main()
