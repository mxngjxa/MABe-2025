# Mouse Behavior Detection: Comprehensive EDA Notebook

## Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 120

## File Paths
base_path = Path("data/MABe-mouse-behavior-detection")
train_csv = base_path / "train.csv"
test_csv = base_path / "test.csv"

# Create figures output directory
figures_dir = Path("eda/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
print(f"Figures will be saved to: {figures_dir}")

## Load CSV Data
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

## Data Overview
print(train.head())
print(test.head())

print("Train Info:")
train.info()
print("\nTest Info:")
test.info()

## Missing and Duplication Analysis
print("Missing values (train):")
print(train.isnull().sum()[train.isnull().sum() > 0])
print("Duplicated rows (train):", train.duplicated().sum())

## Data Types and Summary Stats
print("Data types:")
print(train.dtypes.value_counts())
print("Numeric summary:")
print(train.describe())
print("Categorical summary:")
print(train.describe(include="object"))

## Distribution Analysis (Univariate)

for col in ['frames_per_second', 'video_duration_sec', 'video_width_pix', 'video_height_pix', 'arena_width_cm', 'arena_height_cm']:
    plt.figure()
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / f"dist_{col}.png", dpi=300, bbox_inches='tight')
#    plt.show()
    plt.close()

for col in ['arena_shape', 'arena_type', 'tracking_method']:
    plt.figure()
    sns.countplot(y=col, data=train, order=train[col].value_counts().index)
    plt.title(f"{col} Counts")
    plt.tight_layout()
    plt.savefig(figures_dir / f"count_{col}.png", dpi=300, bbox_inches='tight')
#    plt.show()
    plt.close()

## Mouse Characteristics

mouse_strain_cols = [f"mouse{i}_strain" for i in range(1, 5)]
for col in mouse_strain_cols:
    if col in train.columns:
        plt.figure()
        sns.countplot(y=col, data=train, order=train[col].value_counts().index)
        plt.title(f"{col} Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{col}_distribution.png", dpi=300, bbox_inches='tight')
    #    plt.show()
        plt.close()

mouse_sex_cols = [f"mouse{i}_sex" for i in range(1, 5)]
for col in mouse_sex_cols:
    if col in train.columns:
        plt.figure()
        sns.countplot(y=col, data=train, order=train[col].value_counts().index)
        plt.title(f"{col} Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{col}_distribution.png", dpi=300, bbox_inches='tight')
    #    plt.show()
        plt.close()

## Feature Correlations

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(14, 10))
sns.heatmap(train[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Numeric Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(figures_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

## Body Parts & Behaviors (from tracking and annotation)

import ast

if "body_parts_tracked" in train.columns:
    bp_sample = ast.literal_eval(str(train.loc[train["body_parts_tracked"].first_valid_index(), "body_parts_tracked"]))
    print("Example body parts tracked:", bp_sample)

if "behaviors_labeled" in train.columns:
    bl_sample = ast.literal_eval(str(train.loc[train["behaviors_labeled"].first_valid_index(), "behaviors_labeled"]))
    print("Example behaviors labeled:", bl_sample[:10])

## Grouped Analysis by Lab/Experiment

lab_ct = train["lab_id"].value_counts()
plt.figure()
sns.barplot(x=lab_ct.values, y=lab_ct.index, orient="h")
plt.title("Lab Video Counts")
plt.xlabel("Videos")
plt.ylabel("Lab ID")
plt.tight_layout()
plt.savefig(figures_dir / "lab_video_counts.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

for col in ["arena_shape", "arena_type", "tracking_method"]:
    cross = pd.crosstab(train[col], train["lab_id"])
    plt.figure(figsize=(14, 8))
    sns.heatmap(cross, cmap="crest", annot=False)
    plt.title(f"{col} by Lab ID")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{col}_by_lab.png", dpi=300, bbox_inches='tight')
#    plt.show()
    plt.close()

## Sample Tracking and Annotation Data

example_tracking_file = list((base_path / "train_tracking").rglob("*.parquet"))[0]
tracking_df = pd.read_parquet(example_tracking_file)
print("Tracking shape:", tracking_df.shape)
print(tracking_df.head())

example_annotation_file = list((base_path / "train_annotation").rglob("*.parquet"))[0]
annotation_df = pd.read_parquet(example_annotation_file)
print("Annotation shape:", annotation_df.shape)
print(annotation_df.head())

## Time Series Visualization: Mouse Movement (from tracking data)
plt.figure(figsize=(15, 6))
sns.lineplot(x="video_frame", y="x", data=tracking_df, label="X coord")
sns.lineplot(x="video_frame", y="y", data=tracking_df, label="Y coord")
plt.title("Mouse Movement Over Frames")
plt.xlabel("Frame")
plt.ylabel("Position")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "mouse_movement_timeseries.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

## Behavior Distribution (from annotation data if possible)
if 'behavior' in annotation_df.columns:
    plt.figure()
    ann_counts = annotation_df["behavior"].value_counts()
    sns.barplot(x=ann_counts.values, y=ann_counts.index)
    plt.title("Behavior Distribution in Sample Annotation")
    plt.xlabel("Count")
    plt.ylabel("Behavior")
    plt.tight_layout()
    plt.savefig(figures_dir / "behavior_distribution.png", dpi=300, bbox_inches='tight')
#    plt.show()
    plt.close()

## Data Quality & Feature Engineering Suggestions

print("==== Data Quality Observations ====")
missing_summary = train.isnull().sum()
print(missing_summary[missing_summary > 0])
print("\nConsider imputing, dropping or flagging columns with >50% missingness.")

zero_var_cols = [col for col in numeric_cols if train[col].std() == 0]
print("Zero variance columns:", zero_var_cols)

## Next Steps & Recommendations

print("==== Recommendations ====")
print("- Explore inter-mouse distances/relationships with tracking data.")
print("- Analyze temporal patterns in annotated behaviors.")
print("- Create transition matrices for behavioral states.")
print("- Investigate lab-specific protocols as possible confounders.")
print("- Engineer features from movement trajectories for behavior prediction/modeling.")
print("- Profile rare experimental conditions and stratify by arena/experiment metadata.")

## Save Cleaned and Profiled Data (optional demo)
train_clean = train.dropna(axis=1, thresh=train.shape[0] * 0.5)
train_clean.to_csv("eda/train_cleaned.csv", index=False)
print("Saved cleaned train data to eda/train_cleaned.csv.")

## End of Notebook

print(f"\nComprehensive EDA complete.")
print(f"All figures saved to: {figures_dir.absolute()}")
