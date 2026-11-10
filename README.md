# MineToday 2026

## The problem

Intelligo has 17 learning modules. The task is to rank them for each user using assessment answers and chat history. The top five positions matter most. Submissions are scored with mean NDCG@5 and linear relevance gains.

## Data

The six competition inputs are in [`data/raw/`](data/raw/).

| File | What it contains |
| --- | --- |
| `train_relevance.csv` | Relevance scores for 17 modules across 4,000 users |
| `test.csv` | The 1,000 users whose module scores must be predicted |
| `chat_history.csv` | 14,573 chat messages |
| `user_assessments.csv` | Assessment results for 5,000 users |
| `modules_catalog.csv` | Module names, syllabi, and prerequisite levels |
| `sample_submission.csv` | The required columns and user order |

The source project also has a `dataset/` directory for older pipelines. It was left out because its module catalog differs from `data/raw/modules_catalog.csv`.

## What we built

[`notebooks/psi-1_Notebook.ipynb`](notebooks/psi-1_Notebook.ipynb) cleans malformed assessment answers and builds user and module features from the structured fields and text. It uses five folds stratified by each user's most relevant module.

Six model families share those folds. LightGBM ListMLE learns a full ranking, while XGBoost learns pairwise preferences. CatBoost and Ridge supply different model biases. Another branch fits one LightGBM regressor per module, and a two-stage LightGBM model uses first-stage scores as context. A LightGBM ListMLE meta-model combines their out-of-fold predictions. The notebook then applies its fixed user-score adjustment.

The resulting file is [`submissions/submission_det6.csv`](submissions/submission_det6.csv). Before writing it, the notebook checks its columns, user order, and missing values against the sample submission. The [final-round rules](docs/GUDEBOOK_FINAL.md) are kept alongside the notebook.

## Findings from the final presentation

The team called this pipeline MAESTRO. The presentation reports an out-of-fold NDCG@5 of 0.713 for the personalized ranking, compared with 0.396 for a popularity baseline that gives everyone the same five modules. These are presentation results, not scores recalculated from this repository.

Individual assessment skills carried more module-specific information than a single average assessment score. Chat was especially useful when it mentioned a module or matched its catalog text. Recent messages sometimes pointed to a different interest than the full chat history, which is why the feature set keeps both. The presentation reports that 93.7 percent of chats follow a recurring template. That helps explain why word and character TF-IDF, explicit mentions, and keyword matches were useful on this dataset. It also helps explain why a more complex text model was not automatically better in the team's experiments.

The text affinity feature uses Ridge predictions made within the training folds. This matters because fitting the feature on the same relevance labels later used for validation would leak the answer. In the presentation's feature analysis, text affinity features account for roughly 63 percent of the mean SHAP magnitude among the 15 displayed features. That figure describes the plotted features, not the entire model.

Performance varied with available chat history. The presentation reports NDCG@5 of 0.640 for users without chat, 0.708 for users with one or two messages, and 0.726 for users with at least three. The lower score for the no-chat group is a useful warning when interpreting recommendations for new or quiet users.

Rerunning the notebook requires the Python ML libraries it imports and can overwrite the saved submission in this copy.
