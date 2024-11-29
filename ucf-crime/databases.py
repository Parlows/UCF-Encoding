import numpy as np
import os

# Utils
from my_utils import distances
from env import *

# Milvus
from pymilvus import MilvusClient

# QDrant
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# Chroma
#import chromadb

possible_databases = [
    'local',
    'milvus',
    'qdrant',
]

class DatabaseHandler:
    def __init__(self, encoder_params:dict):
        raise NotImplementedError
    
    def __enter__(self):
        raise NotImplementedError
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def upload_embedding(self):
        """Uploads embedding to the database
        """
        raise NotImplementedError

class DatabaseBuilder:

    def build(database_name, encoder_params, *args, **kwargs) -> DatabaseHandler:
        """Returns the database's handler object corresponding to the specified name.

        Parameters
        ----------
        database_name : str
            The name of the database. Can be one of the specified in the \'databases.possible_databases\' variable.

        Returns
        -------
        DatabaseHandler
            The database's handler object.

        Raises
        ------
        TypeError
            If the specified database handler is not implemented.
        """

        match database_name:
            case 'local':
                return LocalDatabase(encoder_params, *args, **kwargs)
            case 'milvus':
                return MilvusDatabase(encoder_params, *args, **kwargs)
            case 'qdrant':
                return QDrantDatabase(encoder_params, *args, **kwargs)
            case _:
                raise TypeError(f'TypeError: Encoder {database_name} not found among implemented. Please, use one of the following: {possible_databases}.')

class LocalDatabase(DatabaseHandler):

    def __init__(self, encoder_params:dict, save_path:str='/home/pablo/Documents/TFM/ucf-crime/clip_embs'):
        """This database saves the embeddings in NumPy .npy file in the specified path.

        Parameters
        ----------
        encoder_params : dict
            The parameters defined by the encoder.
        save_path : str, optional
            The path to save the embeddings, by default '/home/pablo/Documents/TFM/ucf-crime/clip_embs'
        """

        # Get encoder params
        self.encoder_name = encoder_params['model_name']
        self.emb_size = encoder_params['embedding_size']
        self.emb_list = encoder_params['embedding_list']
        self.distance = encoder_params['distance']

        self.save_path = save_path

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def upload_embedding(self, id:int, emb:np.ndarray, metadata:dict):
        """Save the embedding to the file

        Parameters
        ----------
        id : int
            The id of the clip.
        emb : np.ndarray
            The embedding (or a list of embeddings).
        metadata : dict
            The metadata to store alongside the embedding.
        """
        #np_file_name = str(id) + '_' + metadata['video'] + '_' + str(metadata['start_frame']) + '-' + str(metadata['end_frame']) + '.npy'
        if self.emb_list:
            for i, vector in enumerate(emb):
                np_file_name = f'{id:06d}-{i}_{metadata["video"]}_{metadata["start_frame"]}-{metadata["end_frame"]}.npy'
                np.save(os.path.join(self.save_path, np_file_name), vector)
        else:
            np_file_name = f'{id:06d}_{metadata["video"]}_{metadata["start_frame"]}-{metadata["end_frame"]}.npy'
            np.save(os.path.join(self.save_path, np_file_name), emb)

class MilvusDatabase(DatabaseHandler):

    def __init__(self, encoder_params:dict,
                 host:str=DATABASE_HOST,
                 port:int=MILVUS_PORT,
                 rewrite:bool=True,
                 token:str='root:Milvus'):
        """Milvus Database handler.

        Parameters
        ----------
        encoder_params : dict
            The parameters defined by the encoder.
        host : str, optional
            The host where the database is running, by default 'localhost'
        port : int, optional
            The port where the database is running, by default 19530
        rewrite : bool, optional
            Whether to delete any previous collection with the same name, by default True
        token : _type_, optional
            Token to acces Milvus database, by default 'root:Milvus'
        """
        
        # Get encoder params
        self.encoder_name = encoder_params['model_name']
        self.emb_size = encoder_params['embedding_size']
        self.emb_list = encoder_params['embedding_list']
        
        # Connect client
        self.client = MilvusClient(uri=f'http://{host}:{port}', token=token)

        # Name the collection
        collection_name = f'ucf{self.encoder_name}'

        # Check if collection exists
        if (not self.client.has_collection(collection_name=collection_name)) or rewrite:
            
            # Drop collection
            self.client.drop_collection(collection_name=collection_name)
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.emb_size,
                primary_field_name='id',
                id_type='int',
                vector_field_name='vector',
                metric_type='COSINE',
                auto_id=self.emb_list,
                timeout=None,
                schema=None,
                index_params=None
            )
        
        self.collection_name = collection_name

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def close(self):
        self.client.close()

    def upload_embedding(self, id:int, emb:np.ndarray, metadata:dict):
        """Upload the embedding to the database.

        Parameters
        ----------
        id : int
            The id of the clip.
        emb : np.ndarray
            The embedding (or a list of embeddings).
        metadata : dict
            The metadata to store alongside the embedding.
        """
        if self.emb_list:
            data = [{'vector':vector, **metadata} for vector in emb]
        else:
            data = metadata
            data['id'] = id
            data['vector'] = emb

        self.client.insert(collection_name=self.collection_name, data=data)

class QDrantDatabase(DatabaseHandler):

    def __init__(self, encoder_params:dict, host:str=DATABASE_HOST,
                 port:int=QDRANT_PORT,
                 rewrite:bool=True):
        """QDrant Database handler.

        Parameters
        ----------
        encoder_params : dict
            The parameters defined by the encoder.
        host : str, optional
            _The host where the database is running, by default 'localhost'
        port : int, optional
            The port where the database is running, by default 6333
        rewrite : bool, optional
            Whether to delete any previous collection with the same name, by default True
        """

        # Get encoder params
        self.encoder_name = encoder_params['model_name']
        self.emb_size = encoder_params['embedding_size']
        self.emb_list = encoder_params['embedding_list']
        
        # Connect to client
        self.client = QdrantClient(host=host, port=port)

        # Name the collection
        collection_name = f'ucf{self.encoder_name}'

        # Create collection
        if (not self.client.collection_exists(collection_name=collection_name)):
            
            print("Collection do not exist")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.emb_size,
                    distance=Distance.COSINE
                ),
            )
        elif rewrite:
            print(f"[DB_HAND]: Collection '{collection_name}' do exist but will be rewritten")
            self.client.delete_collection(collection_name=collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.emb_size,
                    distance=Distance.COSINE
                ),
            )

        self.collection_name = collection_name

    def __enter__(self):
        pass

    def __exit__(self):
        self.client.close()

    def close(self):
        self.client.close()

    def upload_embedding(self, id:int, emb:np.ndarray, metadata:dict):
        """Upload the embedding to the database.

        Parameters
        ----------
        id : int
            The id of the clip.
        emb : np.ndarray
            The embedding (or a list of embeddings).
        metadata : dict
            The metadata to store alongside the embedding.
        """
        if self.emb_list:
            points = [PointStruct(
                id=id,
                vector=vector.tolist(),
                payload=metadata
            ) for vector in emb]
        else:
            points = [
                PointStruct(
                    id=id,
                    vector=emb.tolist(),
                    payload=metadata)]
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
"""
class ChromaDatabase(DatabaseHandler):

    def __init__(self, encoder_params:dict,
                 host:str=DATABASE_HOST,
                 port:int=CHROMA_PORT,
                 rewrite:bool=True):
        
        # Get encoder params
        self.encoder_name = encoder_params['model_name']
        self.emb_size = encoder_params['embedding_size']
        self.emb_list = encoder_params['embedding_list']

        # Connect to database
        self.client = chromadb.HttpClient(host=host, port=port)
        
        # Name the collection
        self.collection_name = f'ucf{self.encoder_name}'

        # Create connection
        if rewrite:
            if self.collection in [col.name for col in self.client.list_collections()]:
                self.client.delete_collection(name=self.collection)
            else:
                self.client.create_collection(name=self.collection_name)
        else:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def close(self):
        pass

    def upload_embedding(self, id:int, emb:np.ndarray, metadata:dict):

        if self.emb_list:
            data = emb
        else:
            data = [emb]

        self.collection.add(ids=[id],
                            embeddings=data,
                            metadatas=metadata)
"""
