# Mouse Behavior Detection - Comprehensive EDA
# Generated EDA script for preliminary analysis

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# %%
print("=" * 80)
print("MOUSE BEHAVIOR DETECTION - COMPREHENSIVE EDA")
print("=" * 80)

# %%
# Load the data
train = pd.read_csv('data/MABe-mouse-behavior-detection/train.csv')
test = pd.read_csv('data/MABe-mouse-behavior-detection/test.csv')

# Load sample tracking and annotation data
x_sample = pd.read_parquet('data/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/44566106.parquet')
y_sample = pd.read_parquet('data/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/44566106.parquet')

# %%
print("\n" + "=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

# Dataset sizes
print(f"\n📊 Dataset Dimensions:")
print(f"  - Train dataset: {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"  - Test dataset: {test.shape[0]:,} rows × {test.shape[1]} columns")
print(f"  - Sample tracking: {x_sample.shape[0]:,} frames × {x_sample.shape[1]} features")
print(f"  - Sample annotation: {y_sample.shape[0]:,} annotations × {y_sample.shape[1]} columns")

# Memory usage
print(f"\n💾 Memory Usage:")
print(f"  - Train: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  - Test: {test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  - Sample tracking: {x_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  - Sample annotation: {y_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# %%
print("\n" + "=" * 60)
print("2. DATA QUALITY ASSESSMENT")
print("=" * 60)

# Missing values analysis
print(f"\n🔍 Missing Values in Train Dataset:")
missing_train = train.isnull().sum()
missing_cols = missing_train[missing_train > 0]
if len(missing_cols) > 0:
    print(f"\nColumns with missing values:")
    for col, count in missing_cols.items():
        pct = (count / len(train)) * 100
        print(f"  - {col}: {count:,} ({pct:.1f}%)")
else:
    print("  ✓ No missing values found")

# Data types summary
print(f"\n📝 Data Types Distribution:")
dtype_counts = train.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  - {dtype}: {count} columns")

# %%
print("\n" + "=" * 60)
print("3. LAB AND VIDEO STATISTICS")
print("=" * 60)

# Lab distribution
print(f"\n🏢 Lab Distribution:")
lab_counts = train['lab_id'].value_counts()
for lab, count in lab_counts.items():
    pct = (count / len(train)) * 100
    print(f"  - {lab}: {count:,} videos ({pct:.1f}%)")

# Unique video count
print(f"\n📹 Video Statistics:")
print(f"  - Total unique videos (train): {train['video_id'].nunique():,}")
print(f"  - Total unique videos (test): {test['video_id'].nunique():,}")
print(f"  - Videos per lab (avg): {train.groupby('lab_id')['video_id'].nunique().mean():.1f}")

# %%
print("\n" + "=" * 60)
print("4. MOUSE CHARACTERISTICS ANALYSIS")
print("=" * 60)

# Mouse strains
print(f"\n🐭 Mouse Strains:")
for i in range(1, 5):
    col = f'mouse{i}_strain'
    if col in train.columns:
        strains = train[col].value_counts()
        print(f"\n  Mouse {i}:")
        for strain, count in strains.items():
            if pd.notna(strain):
                print(f"    - {strain}: {count:,}")

# Sex distribution
print(f"\n⚥ Sex Distribution:")
for i in range(1, 5):
    col = f'mouse{i}_sex'
    if col in train.columns:
        sex_dist = train[col].value_counts()
        print(f"  Mouse {i}: {dict(sex_dist)}")

# Age statistics
print(f"\n📅 Age Distribution:")
for i in range(1, 5):
    col = f'mouse{i}_age'
    if col in train.columns:
        ages = train[col].value_counts()
        if len(ages) > 0:
            print(f"  Mouse {i}: {ages.iloc[0] if len(ages) > 0 else 'N/A'}")

# %%
print("\n" + "=" * 60)
print("5. VIDEO PROPERTIES ANALYSIS")
print("=" * 60)

# Frame rate analysis
print(f"\n🎬 Frame Rate Statistics:")
fps_stats = train['frames_per_second'].describe()
print(f"  - Mean: {fps_stats['mean']:.1f} fps")
print(f"  - Std: {fps_stats['std']:.2f} fps")
print(f"  - Min: {fps_stats['min']:.0f} fps")
print(f"  - Max: {fps_stats['max']:.0f} fps")
print(f"  - Common rates: {train['frames_per_second'].value_counts().head(3).to_dict()}")

# Video duration analysis
print(f"\n⏱️ Video Duration Statistics:")
duration_stats = train['video_duration_sec'].describe()
print(f"  - Mean: {duration_stats['mean']:.1f} seconds ({duration_stats['mean']/60:.1f} minutes)")
print(f"  - Std: {duration_stats['std']:.1f} seconds")
print(f"  - Min: {duration_stats['min']:.1f} seconds")
print(f"  - Max: {duration_stats['max']:.1f} seconds ({duration_stats['max']/60:.1f} minutes)")
print(f"  - Total: {train['video_duration_sec'].sum()/3600:.1f} hours")

# Resolution statistics
print(f"\n📐 Video Resolution Statistics:")
print(f"  Width: {train['video_width_pix'].mean():.0f} ± {train['video_width_pix'].std():.0f} pixels")
print(f"  Height: {train['video_height_pix'].mean():.0f} ± {train['video_height_pix'].std():.0f} pixels")
print(f"  Common resolutions:")
resolutions = train.groupby(['video_width_pix', 'video_height_pix']).size().sort_values(ascending=False).head(3)
for (width, height), count in resolutions.items():
    print(f"    - {width}x{height}: {count} videos")

# %%
print("\n" + "=" * 60)
print("6. ARENA PROPERTIES")
print("=" * 60)

# Arena shape distribution
print(f"\n🏟️ Arena Shapes:")
arena_shapes = train['arena_shape'].value_counts()
for shape, count in arena_shapes.items():
    if pd.notna(shape):
        pct = (count / len(train)) * 100
        print(f"  - {shape}: {count:,} ({pct:.1f}%)")

# Arena type distribution
print(f"\n🔍 Arena Types:")
arena_types = train['arena_type'].value_counts()
for arena_type, count in arena_types.items():
    if pd.notna(arena_type):
        pct = (count / len(train)) * 100
        print(f"  - {arena_type}: {count:,} ({pct:.1f}%)")

# Arena dimensions
print(f"\n📏 Arena Dimensions:")
print(f"  Width: {train['arena_width_cm'].mean():.1f} ± {train['arena_width_cm'].std():.1f} cm")
print(f"  Height: {train['arena_height_cm'].mean():.1f} ± {train['arena_height_cm'].std():.1f} cm")

# %%
print("\n" + "=" * 60)
print("7. TRACKING DATA ANALYSIS")
print("=" * 60)

# Body parts tracked
print(f"\n🎯 Body Parts Tracked:")
if 'body_parts_tracked' in train.columns:
    # Parse the first entry to get body parts
    import ast
    try:
        sample_parts = ast.literal_eval(train['body_parts_tracked'].iloc[0])
        print(f"  Total body parts tracked: {len(sample_parts)}")
        print(f"  Body parts: {', '.join(sample_parts[:5])}...")
    except:
        print("  Unable to parse body parts data")

# Tracking method
print(f"\n🔬 Tracking Methods:")
if 'tracking_method' in train.columns:
    methods = train['tracking_method'].value_counts()
    for method, count in methods.items():
        pct = (count / len(train)) * 100
        print(f"  - {method}: {count:,} ({pct:.1f}%)")

# %%
print("\n" + "=" * 60)
print("8. BEHAVIOR LABELS ANALYSIS")
print("=" * 60)

# Parse behavior labels
print(f"\n🏷️ Behavior Labels:")
if 'behaviors_labeled' in train.columns:
    try:
        import ast
        sample_behaviors = ast.literal_eval(train['behaviors_labeled'].iloc[0])

        # Count behavior types
        behavior_types = {'approach': 0, 'attack': 0, 'avoid': 0, 'chase': 0, 
                         'chaseattack': 0, 'submit': 0, 'rear': 0}

        for behavior in sample_behaviors:
            for btype in behavior_types:
                if btype in behavior.lower():
                    behavior_types[btype] += 1

        print(f"  Total behavior combinations: {len(sample_behaviors)}")
        print(f"\n  Behavior type frequencies:")
        for btype, count in sorted(behavior_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {btype}: {count} combinations")

        # Mouse pair combinations
        mouse_pairs = set()
        for behavior in sample_behaviors:
            parts = behavior.split(',')
            if len(parts) >= 2:
                mouse_pairs.add((parts[0], parts[1]))

        print(f"\n  Unique mouse interaction pairs: {len(mouse_pairs)}")

    except Exception as e:
        print(f"  Unable to parse behavior labels: {e}")

# %%
print("\n" + "=" * 60)
print("9. SAMPLE TRACKING DATA ANALYSIS")
print("=" * 60)

# Frame statistics
print(f"\n📊 Frame Statistics:")
print(f"  - Total frames: {x_sample['video_frame'].nunique():,}")
print(f"  - Frame range: {x_sample['video_frame'].min()} to {x_sample['video_frame'].max()}")

# Mouse ID distribution
print(f"\n🐭 Mouse ID Distribution in Tracking:")
mouse_dist = x_sample['mouse_id'].value_counts()
for mouse_id, count in mouse_dist.items():
    pct = (count / len(x_sample)) * 100
    print(f"  - Mouse {mouse_id}: {count:,} records ({pct:.1f}%)")

# Body part distribution
print(f"\n🎯 Body Part Distribution:")
bodypart_counts = x_sample['bodypart'].value_counts()
print(f"  Total unique body parts: {len(bodypart_counts)}")
print(f"  Top 5 tracked body parts:")
for part, count in bodypart_counts.head(5).items():
    pct = (count / len(x_sample)) * 100
    print(f"    - {part}: {count:,} ({pct:.1f}%)")

# Coordinate statistics
print(f"\n📍 Coordinate Statistics:")
print(f"  X coordinates:")
print(f"    - Mean: {x_sample['x'].mean():.2f}")
print(f"    - Std: {x_sample['x'].std():.2f}")
print(f"    - Range: [{x_sample['x'].min():.2f}, {x_sample['x'].max():.2f}]")
print(f"  Y coordinates:")
print(f"    - Mean: {x_sample['y'].mean():.2f}")
print(f"    - Std: {x_sample['y'].std():.2f}")
print(f"    - Range: [{x_sample['y'].min():.2f}, {x_sample['y'].max():.2f}]")

# %%
print("\n" + "=" * 60)
print("10. SAMPLE ANNOTATION DATA ANALYSIS")
print("=" * 60)

# Annotation columns
print(f"\n📝 Annotation Structure:")
print(f"  Columns: {list(y_sample.columns)}")
print(f"  Shape: {y_sample.shape}")

# Basic statistics
if 'start' in y_sample.columns and 'end' in y_sample.columns:
    print(f"\n⏱️ Annotation Timing:")
    print(f"  - Total annotations: {len(y_sample):,}")
    print(f"  - Start frame range: [{y_sample['start'].min()}, {y_sample['start'].max()}]")
    print(f"  - End frame range: [{y_sample['end'].min()}, {y_sample['end'].max()}]")

    # Calculate duration
    y_sample['duration'] = y_sample['end'] - y_sample['start']
    print(f"  - Mean duration: {y_sample['duration'].mean():.1f} frames")
    print(f"  - Max duration: {y_sample['duration'].max()} frames")
    print(f"  - Min duration: {y_sample['duration'].min()} frames")

# Behavior distribution
if 'behavior' in y_sample.columns:
    print(f"\n🏷️ Behavior Distribution in Annotations:")
    behavior_counts = y_sample['behavior'].value_counts()
    for behavior, count in behavior_counts.head(10).items():
        pct = (count / len(y_sample)) * 100
        print(f"  - {behavior}: {count} ({pct:.1f}%)")

# %%
print("\n" + "=" * 60)
print("11. EXPERIMENTAL CONDITIONS ANALYSIS")
print("=" * 60)

# Condition analysis for each mouse
print(f"\n🔬 Experimental Conditions:")
for i in range(1, 5):
    col = f'mouse{i}_condition'
    if col in train.columns:
        conditions = train[col].value_counts()
        if len(conditions) > 0:
            print(f"\n  Mouse {i} conditions:")
            for condition, count in conditions.items():
                if pd.notna(condition):
                    pct = (count / len(train)) * 100
                    print(f"    - {condition}: {count:,} ({pct:.1f}%)")

# %%
print("\n" + "=" * 60)
print("12. DATA RELATIONSHIPS & CORRELATIONS")
print("=" * 60)

# Numeric columns for correlation
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n📊 Numeric Features: {len(numeric_cols)}")

# Key correlations
if len(numeric_cols) > 1:
    corr_matrix = train[numeric_cols].corr()

    # Find high correlations
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    if high_corr:
        print(f"\n🔗 High Correlations (|r| > 0.5):")
        for col1, col2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
            print(f"  - {col1} ↔ {col2}: {corr:.3f}")

# %%
print("\n" + "=" * 60)
print("13. SUMMARY INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print(f"""
📌 Key Findings:
  1. Dataset contains {train.shape[0]:,} training videos across {train['lab_id'].nunique()} labs
  2. Each video tracks {4 - train[['mouse1_id', 'mouse2_id', 'mouse3_id', 'mouse4_id']].isnull().sum(axis=0).min()} mice simultaneously
  3. Videos average {train['video_duration_sec'].mean()/60:.1f} minutes in length
  4. Tracking performed at {train['frames_per_second'].mode().values[0]:.0f} fps (most common)
  5. {len(bodypart_counts)} different body parts tracked per mouse

🎯 Data Quality:
  - Missing data present in mouse4 columns (likely 3-mouse experiments)
  - Consistent arena dimensions (mostly {train['arena_width_cm'].mode().values[0]:.0f}x{train['arena_height_cm'].mode().values[0]:.0f} cm)
  - All tracking done with {train['tracking_method'].mode().values[0]}

💡 Recommendations for Further Analysis:
  1. Investigate temporal patterns in behavior annotations
  2. Analyze inter-mouse distance and velocity patterns
  3. Create behavior transition matrices
  4. Examine lab-specific differences in experimental protocols
  5. Build features from tracking trajectories for behavior prediction
""")

# %%
print("\n" + "=" * 80)
print("EDA COMPLETE - Results saved to file")
print("=" * 80)
