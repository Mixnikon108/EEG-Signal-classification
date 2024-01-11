from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Concatenate, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow as tf

#--- ATCNet -----------------------------------------------------------------
#%%
class AttentionBlock(layers.Layer):
    def __init__(self, key_dim=8, num_heads=2, dropout=0.5, **kwargs):
        """
        Capa de atención que implementa un bloque de atención multi-cabeza.

        Parámetros:
        - key_dim: Dimensión de la clave en la atención multi-cabeza.
        - num_heads: Número de cabezas en la atención multi-cabeza.
        - dropout: Tasa de dropout para la regularización.

        """
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.expanded_axis = 3
        self.dropout = dropout

        # Capas utilizadas en el bloque de atención
        self.LayerNormalization = layers.LayerNormalization(epsilon=1e-6)
        self.MultiHeadAttention = layers.MultiHeadAttention(key_dim=self.key_dim, num_heads=self.num_heads, dropout=self.dropout)
        self.Dropout = layers.Dropout(0.3)
        self.Add = layers.Add()
        self.Lambda = layers.Lambda(lambda x: K.squeeze(x, self.expanded_axis))

    def call(self, net):
        """
        Método que realiza la operación de atención en el tensor de entrada.

        Parámetros:
        - net: Tensor de entrada.

        Returns:
        - Tensor después de aplicar el bloque de atención.

        """
        in_sh = net.shape
        in_len = len(in_sh)

        # Aplanar el tensor si tiene más de 3 dimensiones
        if in_len > 3:
            net = layers.Reshape((in_sh[1], -1))(net)

        # Normalización y atención multi-cabeza
        x = self.LayerNormalization(net)
        x = self.MultiHeadAttention(x, x)
        x = self.Dropout(x)
        net = self.Add([net, x])

        # Revertir la aplanación si es necesario
        if in_len == 3 and len(net.shape) == 4:
            net = self.Lambda(net)
        elif in_len == 4 and len(net.shape) == 3:
            net = layers.Reshape((in_sh[1], in_sh[2], in_sh[3]))(net)

        return net

    def get_config(self):
        """
        Obtiene la configuración de la capa AttentionBlock.

        Returns:
        - config: Configuración de la capa.

        """
        config = super().get_config()
        config.update({
            "key_dim": self.key_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout
        })
        return config


#%%
class ConvBlock(layers.Layer):
    def __init__(self, F1=4, kern_length=64, pool_size=8, D=2, in_chans=22, dropout=0.1, **kwargs):
        """
        Capa que implementa un bloque convolucional utilizado en la red ATCNet.

        Parámetros:
        - F1: Número de filtros en la primera capa convolucional.
        - kern_length: Longitud del kernel en la primera capa convolucional.
        - pool_size: Tamaño del pooling utilizado en las capas de pooling.
        - D: Factor de profundización utilizado en la capa de convolución profunda.
        - in_chans: Número de canales de entrada.
        - dropout: Tasa de dropout para la regularización.

        """
        super().__init__(**kwargs)
        self.F1 = F1
        self.D = D
        self.F2 = self.F1 * self.D
        self.kern_length = kern_length
        self.pool_size = pool_size
        self.in_chans = in_chans
        self.dropout = dropout

        # Definición de capas utilizadas en el bloque convolucional
        self.Conv2D_1 = layers.Conv2D(self.F1, (self.kern_length, 1), data_format='channels_last', use_bias=False, padding='same')
        self.Conv2D_2 = layers.Conv2D(self.F2, (16, 1), data_format='channels_last', use_bias=False, padding='same')
        self.BatchNormalization_1 = layers.BatchNormalization(axis=-1)
        self.BatchNormalization_2 = layers.BatchNormalization(axis=-1)
        self.BatchNormalization_3 = layers.BatchNormalization(axis=-1)
        self.DepthwiseConv2D = layers.DepthwiseConv2D((1, self.in_chans), use_bias=False,
                                        depth_multiplier=self.D,
                                        data_format='channels_last',
                                        depthwise_constraint=max_norm(1.))
        self.Activation_1 = layers.Activation('elu')
        self.Activation_2 = layers.Activation('elu')
        self.AveragePooling2D_1 = layers.AveragePooling2D((8, 1), data_format='channels_last')
        self.AveragePooling2D_2 = layers.AveragePooling2D((self.pool_size, 1), data_format='channels_last')
        self.Dropout_1 = layers.Dropout(self.dropout)
        self.Dropout_2 = layers.Dropout(self.dropout)
        
    def call(self, input_layer):
        """
        Método que realiza la operación del bloque convolucional en el tensor de entrada.

        Parámetros:
        - input_layer: Tensor de entrada.

        Returns:
        - Tensor después de aplicar el bloque convolucional.

        """
        # Primera capa convolucional
        block1 = self.Conv2D_1(input_layer)
        block1 = self.BatchNormalization_1(block1)
        
        # Capa de convolución profunda
        block2 = self.DepthwiseConv2D(block1)
        block2 = self.BatchNormalization_2(block2)
        block2 = self.Activation_1(block2)
        block2 = self.AveragePooling2D_1(block2)
        block2 = self.Dropout_1(block2)
        
        # Segunda capa convolucional
        block3 = self.Conv2D_2(block2)
        block3 = self.BatchNormalization_3(block3)
        block3 = self.Activation_2(block3)
        block3 = self.AveragePooling2D_2(block3)
        block3 = self.Dropout_2(block3)
        return block3

    def get_config(self):
        """
        Obtiene la configuración de la capa ConvBlock.

        Returns:
        - config: Configuración de la capa.

        """
        config = super().get_config()
        config.update({
            "F1": self.F1,
            "kern_length": self.kern_length,
            "pool_size": self.pool_size,
            "D": self.D,
            "in_chans": self.in_chans,
            "dropout": self.dropout
        })
        return config


#%%
class TCNBlock(layers.Layer):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, activation='relu', **kwargs):
        """
        Capa que implementa un bloque Temporal Convolutional Network (TCN) utilizado en la red ATCNet.

        Parámetros:
        - input_dimension: Dimensión de la entrada.
        - depth: Profundidad del bloque TCN.
        - kernel_size: Tamaño del kernel en las capas convolucionales.
        - filters: Número de filtros en las capas convolucionales.
        - dropout: Tasa de dropout para la regularización.
        - activation: Función de activación utilizada en las capas convolucionales.

        """
        super().__init__(**kwargs)
        self.input_dimension = input_dimension
        self.depth = depth
        self.kernel_size = kernel_size
        self.filters = filters
        self.dropout = dropout
        self.activation = activation
        
        # Definición de capas utilizadas en el bloque TCN
        self.Conv1D_1 = layers.Conv1D(self.filters, kernel_size=self.kernel_size, dilation_rate=1, activation='linear', padding='causal', kernel_initializer='he_uniform')
        self.Conv1D_2 = layers.Conv1D(self.filters, kernel_size=self.kernel_size, dilation_rate=1, activation='linear', padding='causal', kernel_initializer='he_uniform')
        self.Conv1D_3 = layers.Conv1D(self.filters, kernel_size=1, padding='same')
        self.BatchNormalization_layers = [layers.BatchNormalization() for _ in range(4)]
        self.Activation_layers = [layers.Activation(self.activation) for _ in range(6)]
        self.Dropout_layers = [layers.Dropout(self.dropout) for _ in range(4)]
        self.Add_layers = [layers.Add() for _ in range(3)]
        
        self.Conv1D_layers_depth = []
        for i in range(self.depth - 1):
            Layer = layers.Conv1D(self.filters, kernel_size=self.kernel_size, dilation_rate=2**(i + 1), activation='linear', padding='causal', kernel_initializer='he_uniform')
            self.Conv1D_layers_depth.append([Layer] * 2)

    def call(self, input_layer):
        """
        Método que realiza la operación del bloque TCN en el tensor de entrada.

        Parámetros:
        - input_layer: Tensor de entrada.

        Returns:
        - Tensor después de aplicar el bloque TCN.

        """
        # Primera capa convolucional
        block = self.Conv1D_1(input_layer)
        block = self.BatchNormalization_layers[0](block)
        block = self.Activation_layers[0](block)
        block = self.Dropout_layers[0](block)
        
        # Segunda capa convolucional
        block = self.Conv1D_2(block)
        block = self.BatchNormalization_layers[1](block)
        block = self.Activation_layers[1](block)
        block = self.Dropout_layers[1](block)
        
        # Añadir entrada original (skip connection)
        if self.input_dimension != self.filters:
            conv = self.Conv1D_3(input_layer)
            added = self.Add_layers[0]([block, conv])
        else:
            added = self.Add_layers[1]([block, input_layer])
        
        # Aplicar función de activación
        out = self.Activation_layers[2](added)

        # Capas convolucionales adicionales en profundidad
        for i in range(self.depth - 1):
            block = self.Conv1D_layers_depth[i][0](out)
            block = self.BatchNormalization_layers[2](block)
            block = self.Activation_layers[3](block)
            block = self.Dropout_layers[2](block)
            block = self.Conv1D_layers_depth[i][1](block)
            block = self.BatchNormalization_layers[3](block)
            block = self.Activation_layers[4](block)
            block = self.Dropout_layers[3](block)
            
            # Añadir salida anterior (skip connection)
            added = self.Add_layers[2]([block, out])
            out = self.Activation_layers[5](added)

        return out

    def get_config(self):
        """
        Obtiene la configuración de la capa TCNBlock.

        Returns:
        - config: Configuración de la capa.

        """
        config = super().get_config()
        config.update({
            "input_dimension": self.input_dimension,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "dropout": self.dropout,
            "activation": self.activation
        })
        return config
    

#%%
def create_ATCNet(in_chans=22, in_samples=1125, n_classes=4, n_windows=5, eegn_F1=16, 
                  eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3, tcn_depth=2, 
                  tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3, tcn_activation='elu'):
    """
    Crea un modelo de red ATCNet utilizando Keras.

    Parámetros:
    - in_chans: Número de canales de entrada.
    - in_samples: Número de muestras de entrada.
    - n_classes: Número de clases de salida.
    - n_windows: Número de ventanas deslizantes.
    - eegn_F1: Número de filtros en la primera capa de convolución.
    - eegn_D: Factor de expansión en la primera capa de convolución.
    - eegn_kernelSize: Tamaño del kernel en la primera capa de convolución.
    - eegn_poolSize: Tamaño de la ventana de pooling en la primera capa de convolución.
    - eegn_dropout: Tasa de dropout en la primera capa de convolución.
    - tcn_depth: Profundidad de la red TCN.
    - tcn_kernelSize: Tamaño del kernel en las capas de TCN.
    - tcn_filters: Número de filtros en las capas de TCN.
    - tcn_dropout: Tasa de dropout en las capas de TCN.
    - tcn_activation: Función de activación en las capas de TCN.

    Returns:
    - Modelo de la red ATCNet.

    """
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)

    # Primer bloque convolucional
    block1 = ConvBlock(F1=eegn_F1, D=eegn_D, 
                    kern_length=eegn_kernelSize, pool_size=eegn_poolSize,
                    in_chans=in_chans, dropout=eegn_dropout)(input_2)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    sw_concat = []
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        # Bloque de atención
        block2 = AttentionBlock()(block2)

        # Bloque de red TCN
        block3 = TCNBlock(input_dimension=eegn_F1 * eegn_D, depth=tcn_depth,
                          kernel_size=tcn_kernelSize, filters=tcn_filters,
                          dropout=tcn_dropout, activation=tcn_activation)(block2)
        # Obtener mapas de características de la última secuencia
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        if i == 0:
            sw_concat = block3
        else:
            sw_concat = Concatenate()([sw_concat, block3])

    # Capa de salida
    output = Dense(n_classes, kernel_constraint=max_norm(.25), activation='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=output)


#--- Spectral Transformer -----------------------------------------------------------------
#%%
def calculate_psd(data, fs=250):
    """
    Calcula el espectro de potencia (PSD) para cada canal en cada muestra de un conjunto de datos EEG.

    Parameters:
    - data (numpy.ndarray): Datos EEG en formato (n_muestras, n_canales, n_muestras_por_canal).
    - fs (int): Frecuencia de muestreo en Hz.

    Returns:
    - result_data (numpy.ndarray): Array 3D que contiene los PSD calculados.
    """
    # Calcular la longitud de la ventana de Fourier
    size_signal = data.shape[2]
    nfft = 2**int(np.ceil(np.log2(size_signal)))

    # Inicializar una lista para almacenar los resultados del PSD
    result_data = []

    # Iterar sobre cada muestra en el conjunto de datos
    for i in tqdm(range(data.shape[0])):  
        # Inicializar una lista para almacenar los PSD de cada canal en la muestra
        list_psd = []

        # Iterar sobre cada canal en la muestra
        for j in range(data.shape[1]):  
            # Obtener los datos del canal actual
            channel_data = data[i, j, :]

            # Calcular el PSD utilizando la función welch de SciPy
            _, psd = welch(channel_data, fs=fs, nperseg=nfft, noverlap=0, nfft=nfft, scaling='density')

            # Agregar el PSD calculado a la lista
            list_psd.append(psd)

        # Agregar la lista de PSD de la muestra actual a los resultados
        result_data.append(list_psd)

    # Convertir los resultados a un array numpy y devolverlos
    return np.array(result_data)

#%%
class PreLNEncoder(layers.Layer):
    """
    Implementación de una capa del codificador con atención y proyecciones densas para el modelo Spectral Transformer.

    Parameters:
    - dense_dim (tuple): Dimensiones para las capas densas.
    - num_heads (int): Número de cabezas para la atención multi-cabeza.
    - head_size (int): Tamaño de la cabeza para la atención multi-cabeza.
    - kernel_regularizer: Regularización del kernel para las capas densas.
    - kwargs: Argumentos adicionales para la capa.

    """
    def __init__(self, dense_dim, num_heads, head_size, kernel_regularizer, **kwargs):
        super().__init__(**kwargs)
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.head_size = head_size
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim[0], activation="relu", kernel_regularizer=kernel_regularizer),
             layers.Dense(dense_dim[1], activation="relu", kernel_regularizer=kernel_regularizer),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.add1 = layers.Add()
        self.add2 = layers.Add()

    def call(self, inputs): 
        """
        Método de llamada para la capa PreLNEncoder.

        Parameters:
        - inputs: Datos de entrada.

        Returns:
        - output: Salida de la capa.

        """
        # Normalización y atención
        firstNorm_out = self.layernorm_1(inputs)
        attention_output = self.attention(firstNorm_out, firstNorm_out, attention_mask=None)
        
        # Conexión residual y más normalización
        add1_out = self.add1([inputs, attention_output])
        secondNorm_out = self.layernorm_2(add1_out)
        
        # Proyección densa y otra conexión residual
        dense_output = self.dense_proj(secondNorm_out)
        output = self.add2([add1_out, dense_output])
        
        return output

    def get_config(self):
        """
        Obtiene la configuración de la capa PreLNEncoder.

        Returns:
        - config: Configuración de la capa.

        """
        config = super().get_config()
        config.update({
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "kernel_regularizer": self.kernel_regularizer
        })
        return config
    
#%%
class PositionalEmbedding(layers.Layer):
    """
    Implementación de una capa de embedding posicional para el modelo Spectral Transformer.

    Parameters:
    - kwargs: Argumentos adicionales para la capa.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Construye los pesos de la capa.

        Parameters:
        - input_shape: Forma de los datos de entrada. 
        (batch_size, n_samples_per_channel, n_channels)

        """
        _, sequence_length, embed_dim = input_shape
        self.position_embeddings = self.add_weight(
            shape=(sequence_length, embed_dim),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Método de llamada para la capa PositionalEmbedding.

        Parameters:
        - inputs: Datos de entrada.

        Returns:
        - output: Salida de la capa.

        """
        positions = tf.range(start=0, limit=tf.shape(inputs)[-2], delta=1)
        embedded_positions = tf.expand_dims(self.position_embeddings, 0)
        return inputs + embedded_positions

    def get_config(self):
        """
        Obtiene la configuración de la capa PositionalEmbedding.

        Returns:
        - config: Configuración de la capa.

        """
        config = super().get_config()
        return config
    
#%%
class SpectralTransformer(layers.Layer):
    def __init__(self, dense_dim, n_classes, encoder_dense_dims, kernel_regularizer, dropout=0.5, num_heads=8, head_size=128, num_stacked_transformers=4, **kwargs):
        """
        Implementación de un modelo Spectral Transformer con varias capas de transformers apilados.

        Parameters:
        - dense_dim (tuple): Dimensiones para las capas densas.
        - n_channels (int): Número de canales.
        - encoder_dense_dims (tuple): Dimensiones para las capas densas del encoder.
        - kernel_regularizer: Regularización del kernel para las capas densas.
        - dropout (float): Tasa de dropout.
        - num_heads (int): Número de cabezas para la atención multi-cabeza.
        - head_size (int): Tamaño de la cabeza para la atención multi-cabeza.
        - num_stacked_transformers (int): Número de transformers apilados.
        - kwargs: Argumentos adicionales para la capa.

        """
        super().__init__(**kwargs)
        self.num_stacked_transformers = num_stacked_transformers
        self.positional_embedding = PositionalEmbedding()
        self.dense1 = layers.Dense(dense_dim[0], activation='relu', kernel_regularizer=kernel_regularizer)
        self.dense2 = layers.Dense(dense_dim[1], activation='relu', kernel_regularizer=kernel_regularizer)
        self.dense3 = layers.Dense(n_classes, activation='softmax')
        self.dropout_layers = [layers.Dropout(dropout) for _ in range(num_stacked_transformers)]
        self.preLN_encoders = [PreLNEncoder(dense_dim=encoder_dense_dims, num_heads=num_heads, head_size=head_size, kernel_regularizer=kernel_regularizer) for _ in range(num_stacked_transformers)]
        self.GlobalMaxPooling1D = layers.GlobalMaxPooling1D()

    def call(self, inputs):
        """
        Método de llamada para la capa SpectralTransformer.

        Parameters:
        - inputs: Datos de entrada.

        Returns:
        - x: Salida del modelo.

        """
        x = self.positional_embedding(inputs)
        x = self.dense1(x)

        # Stacked Transformers
        for i in range(self.num_stacked_transformers):
            x = self.dropout_layers[i](x)
            x = self.preLN_encoders[i](x)

        x = self.GlobalMaxPooling1D(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def get_config(self):
        """
        Obtiene la configuración de la capa SpectralTransformer.

        Returns:
        - config: Configuración de la capa.

        """
        config = super().get_config()
        config.update({
            "dense_dim": (self.dense1.units, self.dense2.units),  # assuming dense1 and dense2 are your Dense layers
            "n_channels": None,  # Replace with the actual value or attribute name
            "num_heads": None,  # Replace with the actual value or attribute name
            "encoder_dense_dims": None,  # Replace with the actual value or attribute name
            "kernel_regularizer": None,  # Replace with the actual value or attribute name
            "dropout": self.dropout_layers[0].rate,  # assuming dropout_layers[0] is your first Dropout layer
            "head_size": self.preLN_encoders[0].head_size,  # assuming preLN_encoders[0] is your first PreLNEncoder layer
            "num_stacked_transformers": self.num_stacked_transformers
        })
        return config

#%%
def create_spectral_transformer_model(n_samples, n_channels, num_stacked_transformers):
    """
    Create a Spectral Transformer model.

    Parameters:
    - n_samples (int): Number of time samples in the input data.
    - n_channels (int): Number of channels in the input data.
    - num_stacked_transformers (int): Number of stacked Spectral Transformer layers.

    Returns:
    - keras.Model: Spectral Transformer model.
    """
    # Define the input layer
    inputs = Input(shape=(n_samples, n_channels))

    # Build the model using the SpectralTransformer class
    output = SpectralTransformer(
        dense_dim=(64, 128),
        n_classes=2,
        encoder_dense_dims=(32, 64),
        kernel_regularizer=regularizers.l1(0.0005),
        dropout=0.5,
        num_heads=8,
        head_size=128,
        num_stacked_transformers=num_stacked_transformers
    )(inputs)

    # Create the model
    spectral_trans_model = Model(inputs, output)

    return spectral_trans_model