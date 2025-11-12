# %%
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Iterator
from datasets import Dataset, DatasetDict
import pyarrow as pa
import pyarrow.parquet as pq

# %%
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional

def read_parquet_data_recursive(
    base_path: Union[str, Path],
    file_pattern: str = "*.parquet",
    columns: Optional[List[str]] = None,
    add_metadata: bool = True
) -> pd.DataFrame:
    """
    Recursively read all parquet files from a directory structure.
    
    Parameters:
    -----------
    base_path : str or Path
        Base directory to start recursive search
    file_pattern : str, default "*.parquet"
        Pattern to match files (e.g., "*.parquet", "*.csv")
    columns : list, optional
        Specific columns to read from parquet files
    add_metadata : bool, default True
        Whether to add metadata columns (directory, subdirectory, filename)
        
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with all parquet data and metadata columns
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise ValueError(f"Path does not exist: {base_path}")
    
    # Recursively find all parquet files
    parquet_files = list(base_path.rglob(file_pattern))
    
    if not parquet_files:
        raise ValueError(f"No files matching '{file_pattern}' found in {base_path}")
    
    print(f"Found {len(parquet_files)} parquet files to process...")
    
    # Read and combine all parquet files
    dfs = []
    for parquet_file in parquet_files:
        try:
            # Read the parquet file
            df = pd.read_parquet(parquet_file, columns=columns)
            
            if add_metadata:
                # Get relative path from base directory
                relative_path = parquet_file.relative_to(base_path)
                
                # Extract directory information
                parts = relative_path.parts
                
                # Add metadata columns
                df['source_filename'] = parquet_file.name
                df['source_directory'] = parts[0] if len(parts) > 1 else ''
                df['source_subdirectory'] = parts[1] if len(parts) > 2 else ''
                df['source_full_path'] = str(relative_path)
                
            dfs.append(df)
            
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No parquet files could be successfully read")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Successfully combined {len(dfs)} files into dataframe with {len(combined_df)} rows")
    
    return combined_df


def read_parquet_with_custom_metadata(
    base_path: Union[str, Path],
    max_depth: Optional[int] = None
) -> pd.DataFrame:
    """
    Advanced version that handles arbitrary directory depth.
    
    Parameters:
    -----------
    base_path : str or Path
        Base directory to start recursive search
    max_depth : int, optional
        Maximum directory depth to traverse (None for unlimited)
        
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with flexible metadata columns
    """
    base_path = Path(base_path)
    parquet_files = list(base_path.rglob("*.parquet"))
    
    dfs = []
    for parquet_file in parquet_files:
        try:
            # Check depth limit
            relative_path = parquet_file.relative_to(base_path)
            depth = len(relative_path.parts) - 1  # Subtract filename
            
            if max_depth is not None and depth > max_depth:
                continue
            
            df = pd.read_parquet(parquet_file)
            
            # Add comprehensive metadata
            df['filename'] = parquet_file.name
            df['file_stem'] = parquet_file.stem  # Filename without extension
            
            # Add each directory level as separate column
            parts = relative_path.parent.parts
            for i, part in enumerate(parts):
                df[f'dir_level_{i}'] = part
            
            # Add full relative path
            df['full_relative_path'] = str(relative_path)
            df['depth_level'] = depth
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Skipping {parquet_file}: {e}")
            continue
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# Example usage for your data structure:
# # Basic usage - reads all parquet files recursively
# df = read_parquet_data_recursive('_provided/test_tracking')

# # Access metadata
# print(df[['source_directory', 'source_subdirectory', 'source_filename']].head())

# # Filter by specific category
# adaptable_snail_data = df[df['source_directory'] == 'AdaptableSnail']

# # Advanced usage with dynamic depth handling
# df_advanced = read_parquet_with_custom_metadata('_provided/test_tracking', max_depth=2)

# # See all unique directories
# print(df['source_directory'].unique())


# %%
train = pd.read_csv('data/MABe-mouse-behavior-detection/train.csv')
test = pd.read_csv('data/MABe-mouse-behavior-detection/test.csv')

# %%
print(train.head())
# %%
print(test.head())
# %%
print(train.shape, test.shape)
# %% [markdown]
# actual data itself

# %%
x = pd.read_parquet('data/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/44566106.parquet')
print(x.head())
# %%
print(x.shape)

# %%
y = pd.read_parquet('data/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/44566106.parquet')

# %%
print(y.shape)




# Save all analysis results
with open('eda/inputs/v0.0.4a_output.txt', 'w') as f:
    f.write("=== Mouse Behavior Data Analysis ===\n\n")
    
    # Dataset info
    f.write(f"Train shape: {train.shape}\n")
    f.write(f"Test shape: {test.shape}\n")
    f.write(f"Sample tracking shape: {x.shape}\n")
    f.write(f"Sample annotation shape: {y.shape}\n\n")
    
    # DataFrames
    f.write("Train Head:\n")
    f.write(train.head().to_string())
    f.write("\n\nTest Head:\n")
    f.write(test.head().to_string())
    f.write("\n\nTracking Data Sample:\n")
    f.write(x.head().to_string())
