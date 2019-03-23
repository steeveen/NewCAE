# -*- coding: utf-8 -*-
'''
        司马懿：“善败能忍，然厚积薄发”
                                    ——李叔说的
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
          --┃      ☃      ┃--
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗II━II┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
 @Belong = 'NewCAE'  @MadeBy = 'PyCharm'
 @Author = 'steven'   @DateTime = '2019/3/3 20:43'
'''
from keras import Model
from keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPool3D, AveragePooling3D, Concatenate, \
    Conv3DTranspose, MaxPooling3D, Dropout, concatenate, UpSampling3D, Reshape, GlobalAveragePooling3D, \
    GlobalMaxPooling3D, Dense, Flatten, Flatten
from keras.regularizers import l2
import keras.backend as K
from keras_contrib.layers import SubPixelUpscaling
from keras.utils import plot_model


def DenseFCN3D(input_shape, nb_dense_block=5, growth_rate=16, nb_layers_per_block=4,
               reduction=0.0, dropout_rate=0.0, weight_decay=1E-4, init_conv_filters=48,
               include_top=True, weights=None, input_tensor=None, classes=1, activation='softmax',
               upsampling_conv=128, upsampling_type='deconv', early_transition=False,
               transition_pooling='max', initial_kernel_size=(3, 3, 3), initDis='glorot_normal'):
    imgInput = Input(shape=input_shape)

    with K.name_scope('DenseNetFCN'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block + 1):
                raise ValueError('If `nb_dense_block` is a list, its length must be '
                                 '(`nb_dense_block` + 1)')

            bottleneck_nb_layers = nb_layers[-1]
            rev_layers = nb_layers[::-1]
            nb_layers.extend(rev_layers[1:])
        else:
            bottleneck_nb_layers = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        x = Conv3D(init_conv_filters, initial_kernel_size, kernel_initializer=initDis, padding='same',
                   name='initial_conv3D',
                   use_bias=False, kernel_regularizer=l2(weight_decay))(imgInput)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='initial_bn')(x)
        x = Activation('relu')(x)

        nb_filter = init_conv_filters

        skip_list = []

        out_list = []

        if early_transition:
            x = __transition_block3D(x, nb_filter, initDis, compression=compression, weight_decay=weight_decay,
                                     block_prefix='tr_early', transition_pooling=transition_pooling)

        # Add dense blocks and transition down block
        for block_idx in range(nb_dense_block):
            x, nb_filter = __dense_block3D(x, nb_layers[block_idx], nb_filter, growth_rate, initDis,
                                           dropout_rate=dropout_rate,
                                           weight_decay=weight_decay, block_prefix='dense_%i' % block_idx)
            # Skip connection
            skip_list.append(x)
            # add transition_block
            x = __transition_block3D(x, nb_filter, initDis, compression=compression, weight_decay=weight_decay,
                                     block_prefix='tr_%i' % block_idx, transition_pooling=transition_pooling)
            nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

        # The last dense_block does not have a transition_down_block
        # return the concatenated feature maps without the concatenation of the input
        _, nb_filter, concat_list = __dense_block3D(x, bottleneck_nb_layers, nb_filter, growth_rate, initDis,
                                                    dropout_rate=dropout_rate, weight_decay=weight_decay,
                                                    return_concat_list=True,
                                                    block_prefix='dense_%i' % nb_dense_block)
        skip_list = skip_list[::-1]  # reverse the skip list

        # Add dense blocks and transition up block
        for block_idx in range(nb_dense_block):
            n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

            # upsampling block must upsample only the feature maps (concat_list[1:]),
            # not the concatenation of the input with the feature maps (concat_list[0].
            l = concatenate(concat_list[1:], axis=concat_axis)

            t = __transition_up_block3D(l, nb_filters=n_filters_keep, initDis=initDis, type=upsampling_type,
                                        weight_decay=weight_decay,
                                        block_prefix='tr_up_%i' % block_idx)

            # concatenate the skip connection with the transition block
            x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

            # Dont allow the feature map size to grow in upsampling dense blocks
            x_up, nb_filter, concat_list = __dense_block3D(x, nb_layers[nb_dense_block + block_idx + 1],
                                                           initDis=initDis,
                                                           nb_filter=growth_rate, growth_rate=growth_rate,
                                                           dropout_rate=dropout_rate, weight_decay=weight_decay,
                                                           return_concat_list=True, grow_nb_filters=False,
                                                           block_prefix='dense_%i' % (nb_dense_block + 1 + block_idx))
        if early_transition:
            x_up = __transition_up_block3D(x_up, nb_filters=nb_filter, initDis=initDis, type=upsampling_type,
                                           weight_decay=weight_decay,
                                           block_prefix='tr_up_early')
        if include_top:
            x = Conv3D(classes, (1, 1, 1), activation='linear', padding='same', use_bias=False)(x_up)

            if K.image_data_format() == 'channels_first':
                channel, row, col, hig = input_shape
            else:
                row, col, hig, channel = input_shape

            x = Reshape((row * col * hig, classes))(x)
            x = Activation(activation)(x)
            x = Reshape((row, col, hig, classes))(x)
        else:
            x = x_up

        end = x
        model = Model(imgInput, x, name='fcn-densenet')
        return model


def __transition_up_block3D(ip, nb_filters, initDis, type='deconv', weight_decay=1E-4, block_prefix=None):
    '''Adds an upsampling block. Upsampling operation relies on the the type parameter.

    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines
            type of upsampling performed
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.

    # Returns
        a keras tensor
    '''
    with K.name_scope('TransitionUp'):

        if type == 'upsampling':
            x = UpSampling3D(name=name_or_none(block_prefix, '_upsampling'))(ip)
        elif type == 'subpixel':
            x = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
                       use_bias=False, kernel_initializer=initDis, name=name_or_none(block_prefix, '_conv3D'))(ip)
            x = SubPixelUpscaling(scale_factor=2, name=name_or_none(block_prefix, '_subpixel'))(x)
            x = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
                       use_bias=False, kernel_initializer=initDis, name=name_or_none(block_prefix, '_conv3D'))(x)
        else:
            x = Conv3DTranspose(nb_filters, (3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2),
                                kernel_initializer=initDis, kernel_regularizer=l2(weight_decay),
                                name=name_or_none(block_prefix, '_conv3DT'))(ip)
        return x


def __dense_block3D(x, nb_layers, nb_filter, growth_rate, initDis, bottleneck=False, dropout_rate=None,
                    weight_decay=1e-4, grow_nb_filters=True, return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones

    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming

    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor

        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block3D(x, growth_rate, initDis, bottleneck, dropout_rate, weight_decay,
                                block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __conv_block3D(ip, nb_filter, initDis, bottleneck=False, dropout_rate=None, weight_decay=1e-4, block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and relu),
    and optionally a bottleneck layer.

    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming

     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        output tensor of block
    '''
    with K.name_scope('ConvBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer=initDis, padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay), name=name_or_none(block_prefix, '_bottleneck_conv3D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            x = Activation('relu')(x)

        x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer=initDis, padding='same', use_bias=False,
                   name=name_or_none(block_prefix, '_conv3D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    return x


def __transition_block3D(ip, nb_filter, initDis, compression=1.0, weight_decay=1e-4, block_prefix=None,
                         transition_pooling='max'):
    '''
    Adds a pointwise convolution layer (with batch normalization and relu),
    and an average pooling layer. The number of output convolution filters
    can be reduced by appropriately reducing the compression parameter.

    # Arguments
        ip: input keras tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        compression: calculated as 1 - reduction. Reduces the number
            of feature maps in the transition block.
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter * compression, rows / 2, cols / 2)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows / 2, cols / 2, nb_filter * compression)`
        if data_format='channels_last'.

    # Returns
        a keras tensor
    '''
    with K.name_scope('Transition'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('relu')(x)
        x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer=initDis, padding='same',
                   use_bias=False, kernel_regularizer=l2(weight_decay), name=name_or_none(block_prefix, '_conv3D'))(x)
        if transition_pooling == 'avg':
            x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
        elif transition_pooling == 'max':
            x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

        return x


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None


def dense3DClassify(input_shape=None,
                    depth=40,
                    nb_dense_block=3,
                    growth_rate=12,
                    nb_filter=-1,
                    nb_layers_per_block=-1,
                    bottleneck=False,
                    reduction=0.0,
                    dropout_rate=0.0,
                    weight_decay=1e-4,
                    subsample_initial_block=False,
                    include_top=True,
                    input_tensor=None,
                    pooling=None,
                    classes=10,
                    return_concat_list=False,
                    activation='softmax',
                    transition_pooling='avg', initDis='glorot_normal'):
    with K.name_scope('DenseNet'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block):
                raise ValueError('If `nb_dense_block` is a list, its length must match '
                                 'the number of layers provided by `nb_layers`.')

            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7, 7)
            initial_strides = (2, 2, 7)
        else:
            initial_kernel = (3, 3, 3)
            initial_strides = (1, 1, 1)

        img_input = Input(shape=input_shape)

        x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same', name='initial_conv3D',
                   strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

        if subsample_initial_block:
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='initial_bn')(x)
            x = Activation('relu')(x)
            x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = __dense_block3D(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                           dropout_rate=dropout_rate, weight_decay=weight_decay,
                                           block_prefix='dense_%i' % block_idx, initDis=initDis)
            # add transition_block
            x = __transition_block3D(x, nb_filter, compression=compression, weight_decay=weight_decay,
                                     block_prefix='tr_%i' % block_idx, transition_pooling=transition_pooling,
                                     initDis=initDis)
            nb_filter = int(nb_filter * compression)

        # The last dense_block does not have a transition_block
        x, nb_filter, _ = __dense_block3D(x, final_nb_layer, nb_filter, growth_rate, initDis=initDis, bottleneck=bottleneck,
                                          dropout_rate=dropout_rate, weight_decay=weight_decay,
                                          block_prefix='dense_%i' % (nb_dense_block - 1),return_concat_list=True
                                         )

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='final_bn')(x)
        x = Activation('relu')(x)

        if include_top:
            if pooling == 'avg':
                x = GlobalAveragePooling3D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling3D()(x)
            x = Dense(classes, activation=activation)(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling3D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling3D()(x)
        if return_concat_list:
            model = Model(img_input, outputs=[x]+_, name='densenet3D')
        else:
            model = Model(img_input, outputs=x, name='densenet3D')
        return model


def dense3DSemi(input_shape=None,
                depth=40,
                nb_dense_block=3,
                growth_rate=12,
                nb_filter=-1,
                nb_layers_per_block=-1,
                bottleneck=False,
                reduction=0.0,
                dropout_rate=0.0,
                weight_decay=1e-4,
                subsample_initial_block=False, upsampling_type='deconv',
                include_top=True,
                input_tensor=None,
                pooling=None,
                semi_layers_per_block=2,
                semi_growth_rate=2,
                classes=10,
                activation='softmax',
                transition_pooling='avg', initDis='glorot_normal'):
    encoder = dense3DClassify(input_shape=input_shape,
                              dropout_rate=dropout_rate,
                              nb_dense_block=nb_dense_block,
                              nb_layers_per_block=nb_layers_per_block,
                              growth_rate=growth_rate,
                              classes=classes,
                              return_concat_list=True,
                              activation=activation,
                              transition_pooling=transition_pooling,

                              initDis=initDis)
    with K.name_scope('dense3DSemi'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        i1 = Input(shape=input_shape)
        i2 = Input(shape=input_shape)
        x1 = encoder(i1)[0]
        plot_model(encoder,'forkInsight.png',show_shapes=True)
        concat_list = encoder(i2)[1:]

        x1 = MaxPool3D((2, 2, 2), strides=[2, 2, 2], padding='valid')(x1)
        x1 = Conv3D(640, (1, 1, 1), activation='elu', padding='same')(x1)
        x1 = Flatten()(x1)
        x1 = Dense(512, activation='elu')(x1)
        x1 = Dense(256, activation='elu')(x1)
        x1 = Dense(128, activation='elu')(x1)
        out1 = Dense(1, activation='sigmoid')(x1)

        bottleneck_nb_layers = semi_layers_per_block
        nb_layers = [semi_layers_per_block] * (nb_dense_block + 1)

        for block_idx in range(nb_dense_block-1):
            n_filters_keep = semi_growth_rate * nb_layers[block_idx]
            l = concatenate(concat_list[1:], axis=concat_axis)
            x2 = __transition_up_block3D(l, nb_filters=n_filters_keep, initDis=initDis, type=upsampling_type,
                                         weight_decay=weight_decay,
                                         block_prefix='tr_up_%i' % block_idx)

            # Dont allow the feature map size to grow in upsampling dense blocks
            x2_up, nb_filter, concat_list = __dense_block3D(x2, nb_layers[ block_idx ],
                                                            initDis=initDis,
                                                            nb_filter=nb_filter, growth_rate=growth_rate,
                                                            dropout_rate=dropout_rate, weight_decay=weight_decay,
                                                            return_concat_list=True, grow_nb_filters=False,
                                                            block_prefix='dense_%i' % (nb_dense_block + 1 + block_idx))
        x2 = Conv3D(input_shape[3], (1, 1, 1), activation='linear', padding='same', use_bias=False)(x2_up)
    return Model(inputs=[i1, i2], outputs=[out1, x2])


if __name__ == '__main__':
    from keras.utils import plot_model
    mm = dense3DSemi(input_shape=(32, 32, 32, 3), dropout_rate=0.4, nb_dense_block=5, nb_layers_per_block=9,
                     growth_rate=16,
                     classes=512, activation='elu', initDis='glorot_normal')
    mm.summary()
    plot_model(mm,'semiArc.png',show_shapes=True,rankdir='TB')
