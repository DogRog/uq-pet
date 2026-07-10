"""Utilities to visualize the PET dataset (NER + relations).

Primary function:
  - plot_pet_overview(ner, rel, show=True, figsize=(16,10))

The functions accept dataset splits like the notebook's `ner_dataset['test']`
and `rel_dataset['test']` (list-like objects with fields used below).
"""
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def _extract_basic_stats(ner):
    """Return tag names, tag_counter, sent_lengths, docs_sentences, doc_ner."""
    # try to get tag names from dataset features; fall back to indices
    ner_tag_names = None
    if hasattr(ner, "features") and "ner-tags" in ner.features:
        try:
            ner_tag_names = ner.features["ner-tags"].feature.names
        except Exception:
            ner_tag_names = None

    # Flatten tags
    all_tags = []
    for tags in ner["ner-tags"]:
        all_tags.extend(tags)
    tag_counter = Counter(all_tags)

    sent_lengths = [len(tok) for tok in ner["tokens"]]

    docs_sentences = defaultdict(int)
    for doc in ner[:]["document name"]:
        docs_sentences[doc] += 1

    doc_ner = defaultdict(lambda: defaultdict(int))
    for i in range(len(ner)):
        doc = ner[i]["document name"]
        for tag_id in ner[i]["ner-tags"]:
            doc_ner[doc][tag_id] += 1

    return ner_tag_names, tag_counter, sent_lengths, docs_sentences, doc_ner


def plot_pet_overview(ner, rel, show=True, figsize=(16, 10)):
    """Create the PET dataset overview plots (same layout as the notebook).

    Args:
        ner: dataset split (e.g., ner_dataset['test']) with fields:
             - 'tokens', 'ner-tags', 'document name'
        rel: relation dataset split (not required for all plots)
        show: whether to call `plt.show()`
        figsize: matplotlib figure size

    Returns:
        Matplotlib `fig` object.
    """
    ner_tag_names, tag_counter, sent_lengths, docs_sentences, doc_ner = _extract_basic_stats(ner)

    # default tag names when not available
    if ner_tag_names is None:
        max_tag_id = max(tag_counter.keys()) if tag_counter else 0
        ner_tag_names = [f"tag_{i}" for i in range(max_tag_id + 1)]

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("PET Dataset — NER Corpus Overview", fontsize=16, fontweight='bold')

    # 1. Tag distribution
    tags_sorted = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)
    tag_labels = [ner_tag_names[t[0]] for t in tags_sorted]
    tag_values = [t[1] for t in tags_sorted]
    colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(tag_labels))))
    axes[0, 0].barh(tag_labels, tag_values, color=colors)
    axes[0, 0].set_xlabel("Count")
    axes[0, 0].set_title("NER Tag Distribution")

    # 2. Sentence length distribution
    axes[0, 1].hist(sent_lengths, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0, 1].set_xlabel("Tokens per Sentence")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Sentence Length Distribution\n(mean={np.mean(sent_lengths):.1f}, max={max(sent_lengths)})")
    axes[0, 1].axvline(np.mean(sent_lengths), color='red', linestyle='--', label=f"Mean: {np.mean(sent_lengths):.1f}")
    axes[0, 1].legend()

    # 3. Sentences per document
    doc_sorted = sorted(docs_sentences.values())
    axes[0, 2].hist(doc_sorted, bins=20, color='coral', edgecolor='white', alpha=0.8)
    axes[0, 2].set_xlabel("Sentences per Document")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title(f"Documents: {len(doc_sorted)}\nAvg sentences/doc: {np.mean(doc_sorted):.1f}")

    # 4. Tag distribution by high-level categories (best-effort mapping)
    # Attempt to map B-/I- prefixes into categories similar to the notebook
    category_map = {
        'Actor': ['B-Actor', 'I-Actor'],
        'Activity': ['B-Activity', 'I-Activity'],
        'Activity Data': ['B-Activity Data', 'I-Activity Data'],
        'Further Spec.': ['B-Further Specification', 'I-Further Specification'],
        'XOR Gateway': ['B-XOR Gateway', 'I-XOR Gateway'],
        'AND Gateway': ['B-AND Gateway', 'I-AND Gateway'],
        'Condition Spec.': ['B-Condition Specification', 'I-Condition Specification'],
    }
    # build reverse map from tag name to index
    name_to_idx = {name: i for i, name in enumerate(ner_tag_names)}
    cat_counts = {}
    for cat, tags in category_map.items():
        count = sum(tag_counter.get(name_to_idx.get(t, -1), 0) for t in tags if t in name_to_idx)
        cat_counts[cat] = count
    # If mapping produced zeros (e.g., different tag names), fall back to grouping by top-level labels
    if sum(cat_counts.values()) == 0:
        # simple heuristic: strip B-/I- and count
        stripped = Counter()
        for tid, cnt in tag_counter.items():
            label = ner_tag_names[tid].replace('B-', '').replace('I-', '')
            stripped[label] += cnt
        cats, vals = zip(*sorted(stripped.items(), key=lambda x: x[1], reverse=True))
    else:
        cats, vals = zip(*sorted(cat_counts.items(), key=lambda x: x[1], reverse=True))

    axes[1, 0].pie(vals, labels=cats, autopct='%1.1f%%', startangle=90,
                   colors=plt.cm.Pastel1(np.linspace(0, 1, len(cats))))
    axes[1, 0].set_title("NER Tags by Category")

    # 5. Non-O vs O tag ratio
    o_count = tag_counter.get(0, 0)  # O is commonly index 0
    non_o_count = sum(tag_counter.values()) - o_count
    axes[1, 1].pie([non_o_count, o_count], labels=['Named Entities', 'O (Outside)'],
                   autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'], explode=(0.05, 0))
    axes[1, 1].set_title(f"Entity vs Non-Entity Tokens\n({non_o_count + o_count:,} total tokens)")

    # 6. Top documents by sentence count
    top_docs = sorted(docs_sentences.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_docs:
        doc_labels = [d[0] for d in top_docs]
        doc_vals = [d[1] for d in top_docs]
        axes[1, 2].barh(range(len(doc_labels)), doc_vals, color='mediumseagreen')
        axes[1, 2].set_yticks(range(len(doc_labels)))
        axes[1, 2].set_yticklabels(doc_labels)
        axes[1, 2].set_xlabel("Number of Sentences")
        axes[1, 2].set_title("Top 10 Documents by Sentence Count")
        axes[1, 2].invert_yaxis()
    else:
        axes[1, 2].text(0.5, 0.5, 'No documents available', ha='center')

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_ner_heatmap(ner, show=True):
    """Plot a heatmap of NER tag counts per document using Plotly.

    Returns the Plotly `Figure` object.
    """
    ner_tag_names, tag_counter, sent_lengths, docs_sentences, doc_ner = _extract_basic_stats(ner)
    doc_names_sorted = sorted(doc_ner.keys())
    tag_indices = list(range(1, len(ner_tag_names))) if len(ner_tag_names) > 1 else [0]
    tag_labels_short = [t.replace("B-", "").replace("I-", "") for t in ner_tag_names[1:]] if len(ner_tag_names) > 1 else ner_tag_names

    heatmap_data = np.zeros((len(doc_names_sorted), len(tag_indices)))
    for i, doc in enumerate(doc_names_sorted):
        for j, tid in enumerate(tag_indices):
            heatmap_data[i, j] = doc_ner[doc].get(tid, 0)

    fig = px.imshow(heatmap_data,
                    x=tag_labels_short,
                    y=doc_names_sorted,
                    color_continuous_scale='YlOrRd',
                    aspect='auto',
                    labels=dict(x="NER Tag Type", y="Document", color="Count"))
    fig.update_layout(title="NER Tag Distribution Across Documents (Heatmap)", height=600, width=900)
    return fig
