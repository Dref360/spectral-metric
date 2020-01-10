from .Xception_feature import XceptionFeatures
from .autoencoder import AutoencoderEmbd
from .cnn_autoencoder import CNNAutoencoderEmbd
from .common import NoneGetter
from .tsne import TSNEEmbedding
from .vgg19_feature import VGGFeatures

embeddings = [XceptionFeatures, AutoencoderEmbd, CNNAutoencoderEmbd,NoneGetter, VGGFeatures]
embeddings_dict = {k.get_embedding_name():k for k in embeddings}
