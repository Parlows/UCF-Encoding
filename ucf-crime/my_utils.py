import json
import pandas as pd
from os.path import join

distances = ['DOT', 'COS']

def read_json(json_path:str, json_file:str) -> dict:
    with open(join(json_path, json_file), 'rb') as fp:
        data = json.load(fp)
    return data

def read_uca_as_df(uca_path:str='/media/pablo/358690d7-e500-45fb-b8f8-bc48c6be13e3/Surveillance-Video-Understanding/UCF Annotation/json') -> pd.DataFrame:
    """Get the UCA dataset as a Pandas DataFrame, properly pre-processed.

    Parameters
    ----------
    uca_path : str, optional
        Path to the JSON files of the UCA dataset, by default '/media/pablo/358690d7-e500-45fb-b8f8-bc48c6be13e3/Surveillance-Video-Understanding/UCF Annotation/json'

    Returns
    -------
    pd.DataFrame
        The UCA dataset as a Pandas DataFrame.
    """

    # Read three JSONs
    train_set = read_json(uca_path, 'UCFCrime_Train.json')
    val_set = read_json(uca_path, 'UCFCrime_Val.json')
    test_set = read_json(uca_path, 'UCFCrime_Test.json')

    # Transform each JSON into a Panda's Dataframe
    train_df = pd.DataFrame(train_set).transpose()
    train_df['video'] = train_df.index
    train_df = train_df.reset_index(drop=True)
    val_df = pd.DataFrame(val_set).transpose()
    val_df['video'] = val_df.index
    val_df = val_df.reset_index(drop=True)
    test_df = pd.DataFrame(test_set).transpose()
    test_df['video'] = test_df.index
    test_df = test_df.reset_index(drop=True)

    # Add a column specifying the dataset
    train_df['dataset'] = 'train'
    val_df['dataset'] = 'val'
    test_df['dataset'] = 'test'

    # Merge three dataframes together
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df = df.reset_index(drop=True)

    # Make an entry per sentence
    df = df.explode(['sentences', 'timestamps']).reset_index(drop=True)

    # Create clip duration column
    df['clip_duration'] = df['timestamps'].apply(lambda x: x[1]-x[0])

    # Rename the columns
    df = df.rename(columns={'timestamps':'timestamp', 'sentences':'sentence', 'duration': 'video_duration'})

    # Create class column
    df['class_name'] = df['video'].apply(lambda x: x[:-8])

    # Rename one of the classes
    df = df.replace('Normal_Videos_', 'Normal_Videos')

    # Create anomaly column
    df['anomaly'] = df['class_name'].apply(lambda x: False if x == 'Normal_Videos' else True)

    # Create sentence length column
    df['sentence_length'] = df['sentence'].apply(lambda x: len(x))

    # Return dataframe
    return df