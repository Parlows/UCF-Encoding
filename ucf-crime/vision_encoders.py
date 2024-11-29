import numpy as np
import torch
#from transformers import CLIPModel, AutoProcessor

import os

import finetunedclip.modified_clip.vclip as vclip
from yacs.config import CfgNode as CN
from yacs.config import CfgNode as CN

from PIL import Image

possible_models = [
    'default',
    'random',
    'clip',
    'clip-centroid',
    'vclip',
    'vclipcentroid'
]

class EmbeddingModel:
    def __init__(self):
        raise NotImplementedError

    def get_clip_embedding(self):
        """Generates the embedding of the clip.
        """
        raise NotImplementedError
    
    def get_encoder_params(self) -> dict:
        """Returns the params related with the encoder, to properly configure a database's
        collection.

        Returns
        -------
        dict
            The parameters as a dictionary
        """
        raise NotImplementedError

class EncoderBuilder:

    def build(encoder_name, *args, **kwargs) -> EmbeddingModel:
        """Returns the encoder's object corresponding to the specified name.

        Parameters
        ----------
        encoder_name : str
            The name of the encoder. Can be one of the specified in the \'vision_encoders.possible_models\' variable.

        Returns
        -------
        EmbeddingModel
            The encoder's object.

        Raises
        ------
        TypeError
            If the specified encoder is not implemented.
        """

        match encoder_name:
            case 'default':
                return DefaultEncoder(*args, **kwargs)
            case 'random':
                return RandomEncoder(*args, **kwargs)
            case 'clip':
                return CLIP(*args, **kwargs)
            case 'clip-centroid':
                return CLIPCentroid(*args, **kwargs)
            case 'vclip':
                return VCLIP(*args, **kwargs)
            case 'vclipcentroid':
                return VCLIPCentroid(*args, **kwargs)
            case _:
                raise TypeError(f'TypeError: Encoder {encoder_name} not found among implemented. Please, use one of the following: {possible_models}.')

class DefaultEncoder(EmbeddingModel):

    def __init__(self, defaul_embedding=(1,2,3,4)):
        """This encoder always return the same embedding whenever it is asked to do so.

        Parameters
        ----------
        defaul_embedding : Any, optional
            The embedding which will be returned whenever get_clip_embedding is called. By default, (1,2,3,4).
        """
        self.default_embedding = defaul_embedding

    def get_clip_embedding(self):
        """Returns the embedding specified in the constructor of the class.

        Returns
        -------
        Any
            The embedding specified in the constructor of the class.
        """
        return self.default_embedding
    
    def get_encoder_params(self) -> dict:
        params = {
            'model_name': 'default',
            'embedding_size': len(self.default_embedding),
            'embedding_list': False
        }
        return params

class RandomEncoder(EmbeddingModel):

    def __init__(self, embedding_size:tuple=(768,)):
        """Generates a random embedding of the given size whenever
        get_clip_embedding is called.

        Parameters
        ----------
        embedding_size : tuple, optional
            The desired size for the embedding. By default, (768,).

        Raises
        ------
        TypeError
            If the specified size is not valid.
        """
        if not all(isinstance(i, int) for i in embedding_size):
            raise TypeError(f'TypeError: The specified size for the embedding must be a list of integers')
        else:
            self.embedding_size = embedding_size

    def get_clip_embedding(self, *args, **kwargs):
        """Returns a random embedding of the size specified in the constructor of the class.

        Returns
        -------
        np.ndarray
            A random NumPy array of float values with the size specified in the constructor of
            the class.
        """
        return np.random.rand(*self.embedding_size)
    
    def get_encoder_params(self) -> dict:
        params = {
            'model_name': 'random',
            'embedding_size': self.embedding_size,
            'embedding_list': False
        }
        return params

class CLIP(EmbeddingModel):

    def __init__(self, model_name:str='openai/clip-vit-large-patch14'):
        """Uses HuggingFace's CLIP model to obtain the embeddings of the clips.

        Parameters
        ----------
        model_name : str, optional
            The specific CLIP model from the HuggingFace repository. By default, openai/clip-vit-large-patch14
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CLIPModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except OSError as e:
            print(f'OSError: Model \'{model_name}\' not listed in HuggingFace repository. {e}')

    def _get_img_embedding(self, img:np.ndarray):
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)

        image_features = self.model.get_image_features(**inputs)

        return image_features.cpu().detach().numpy()

    def get_clip_embedding(self, clip_array:list):
        """Generates the embeddings for each frame of the clip. One should be careful with this implementation,
        since it returns a list of embeddings, not an embedding.

        Parameters
        ----------
        clip_array : list
            A list of the frames belonging to the clip.

        Returns
        -------
        np.ndarray
            A NumPy array with all the embeddings of each one of the frames.
        """
        embs_array = np.zeros((len(clip_array), 768))
        for i, frame in enumerate(clip_array):
            embs_array[i] = self._get_img_embedding(frame)
        print(f"[ENCODER]: Encoded {len(embs_array)} embeddings")
        return embs_array
    
    def get_encoder_params(self) -> dict:
        params = {
            'model_name': 'clip',
            'embedding_size': 768,
            'embedding_list': True
        }
        return params
    
class CLIPCentroid(CLIP):

    def get_clip_embedding(self, clip_array:list):
        """Generates the embedding of the clip by averaging the CLIP-generated embeddings
        of the frames.

        Parameters
        ----------
        clip_array : list
            A list of the frames belonging to the clip.

        Returns
        -------
        np.ndarray
            A single embedding which is the average of the embeddings of the clip's frames.
        """
        emb = np.mean(super().get_clip_embedding(clip_array), axis=0)
        return emb
    
    def get_encoder_params(self) -> dict:
        params = {
            'model_name': 'clipcentroid',
            'embedding_size': 768,
            'embedding_list': False
        }
        return params

VCLIP_WEIGHTS_PATH = '/home/pregodon@gaps_domain.ssr.upm.es/TFM/ucf-crime/finetunedclip/weights'
class VCLIP(EmbeddingModel):
    def __init__(self):

        # Apply a default configuration
        _C = CN()
        _C.BASE = ['']
        _C.MODEL = CN()
        _C.MODEL.ARCH = 'ViT-B/32'
        _C.MODEL.WEIGHTS_DIR = VCLIP_WEIGHTS_PATH
        _C.MODEL.RESUME = os.path.join(VCLIP_WEIGHTS_PATH, '100batch_40frames_32.pth')
        _C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = _C.clone()

        self.load()

    def load(self):
        backbone_name = self.config.MODEL.ARCH
        root = self.config.MODEL.WEIGHTS_DIR

        self.model, self.preprocess = vclip.load(name=backbone_name,
                                                 download_root=root,
                                                 jit=False,
                                                 device=self.config.DEVICE)
        
        self.model = self.model.float().cuda()
        checkpoint = torch.load(self.config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']
        self.model.load_state_dict(load_state_dict, strict=False)

        self.device = self.config.DEVICE
    
    def unload(self):
        pass

    def encode_image(self, img):
        # Send image to device
        img = Image.fromarray(img)
        image = self.preprocess(img).unsqueeze(0).to(self.device)

        # Get features
        image_features, attention_weights = self.model.encode_image(image)
        return image_features.cpu().detach().numpy()

    def encode_text(self, text):
        text = vclip.tokenize(text).to(self.device)
        text_features, _ = self.model.encode_text(text)
        return text_features
    
    def get_clip_embedding(self, clip_array:list):
        embs_array = np.zeros((len(clip_array), 512))
        for i, frame in enumerate(clip_array):
            embs_array[i] = self.encode_image(frame)
        print(f"[ENCODER]: Encoded {len(embs_array)} embeddings")
        return embs_array

    def get_encoder_params(self) -> dict:
        params = {
            'model_name': 'vclip',
            'embedding_size': 512,
            'embedding_list': True
        }
        return params

class VCLIPCentroid(VCLIP):

    def get_clip_embedding(self, clip_array:list):
        emb = np.mean(super().get_clip_embedding(clip_array), axis=0)
        return emb
    
    def get_encoder_params(self) -> dict:
        params = {
            'model_name': 'vclipcentroid',
            'embedding_size': 512,
            'embedding_list': False # Means that the get_clip_embedding return a single embedding
        }
        return params
