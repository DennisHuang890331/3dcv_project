import math

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        sr_ratio,
        qkv_bias=False,
        attn_drop_rate=0,
        proj_drop_rate=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads

        self.units = self.num_heads * self.head_dim
        self.sqrt_of_units = math.sqrt(self.head_dim)
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate

        self.q = tf.keras.layers.Dense(self.units, use_bias=self.qkv_bias)
        self.k = tf.keras.layers.Dense(self.units, use_bias=self.qkv_bias)
        self.v = tf.keras.layers.Dense(self.units, use_bias=self.qkv_bias)

        self.attn_drop = tf.keras.layers.Dropout(self.attn_drop_rate)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(
                filters=dim, kernel_size=sr_ratio, strides=sr_ratio, name='sr',
            )
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-05)
           
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(self.proj_drop_rate)

    def call(
        self,
        x,
        H,
        W,
    ):
        get_shape = tf.shape(x)
        B = get_shape[0]
        C = get_shape[2]

        q = self.q(x)
        q = tf.reshape(
            q, shape=(tf.shape(q)[0], -1, self.num_heads, self.head_dim)
        )
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = tf.reshape(x, (B, H, W, C))
            x = self.sr(x)
            x = tf.reshape(x, (B, -1, C))
            x = self.norm(x)

        k = self.k(x)
        k = tf.reshape(
            k, shape=(tf.shape(k)[0], -1, self.num_heads, self.head_dim)
        )
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = self.v(x)
        v = tf.reshape(
            v, shape=(tf.shape(v)[0], -1, self.num_heads, self.head_dim)
        )
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attn = tf.matmul(q, k, transpose_b=True)
        scale = tf.cast(self.sqrt_of_units, dtype=attn.dtype)
        attn = tf.divide(attn, scale)

        attn = tf.nn.softmax(logits=attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, -1, self.units))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "sr_ratio": self.sr_ratio,
                "qkv_bias": self.qkv_bias,
                "attn_drop_rate": self.attn_drop_rate,
                "proj_drop_rate": self.proj_drop_rate
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()    
class MLP(tf.keras.layers.Layer):
    def __init__(self, decode_dim, **kwargs):
        super().__init__(**kwargs)
        self.decode_dim = decode_dim
        self.proj = tf.keras.layers.Dense(self.decode_dim)

    def call(self, x):
        x = self.proj(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decode_dim": self.decode_dim,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class ConvModule(tf.keras.layers.Layer):
    def __init__(self, decode_dim, **kwargs):
        super().__init__(**kwargs)
        self.decode_dim = decode_dim
        self.conv = tf.keras.layers.Conv2D(
            filters=decode_dim, kernel_size=1, use_bias=False
        )
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.activate = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decode_dim": self.decode_dim,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class SegFormerHead(tf.keras.layers.Layer):
    def __init__(self, num_mlp_layers=4, decode_dim=768, num_classes=19, **kwargs):
        super().__init__(**kwargs)
        self.num_mlp_layers = num_mlp_layers
        self.decode_dim = decode_dim
        self.num_classes = num_classes
        self.linear_layers = []
        for _ in range(self.num_mlp_layers):
            self.linear_layers.append(MLP(self.decode_dim))

        self.linear_fuse = ConvModule(self.decode_dim)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.linear_pred = tf.keras.layers.Conv2D(self.num_classes, kernel_size=1)

    def call(self, inputs):
        H = tf.shape(inputs[0])[1]
        W = tf.shape(inputs[0])[2]
        outputs = []

        for x, mlps in zip(inputs, self.linear_layers):
            x = mlps(x)
            x = tf.image.resize(x, size=(H, W), method="bilinear")
            outputs.append(x)

        x = self.linear_fuse(tf.concat(outputs[::-1], axis=3))
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_mlp_layers": self.num_mlp_layers,
                "decode_dim": self.decode_dim,
                "num_classes": self.num_classes,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class DWConv(tf.keras.layers.Layer):
    def __init__(self, filters=768, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dwconv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1,
            padding="same",
            groups=self.filters,
        )

    def call(self, x, H, W):
        get_shape_1 = tf.shape(x)
        x = tf.reshape(x, (get_shape_1[0], H, W, get_shape_1[-1]))
        x = self.dwconv(x)
        get_shape_2 = tf.shape(x)
        x = tf.reshape(
            x, (get_shape_2[0], get_shape_2[1] * get_shape_2[2], get_shape_2[3])
        )
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class Mlp(tf.keras.layers.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop_rate=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.drop_rate = drop_rate

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Dense(hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = tf.keras.layers.Activation("gelu")
        self.fc2 = tf.keras.layers.Dense(out_features)
        self.drop = tf.keras.layers.Dropout(self.drop_rate )

    def call(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "drop_rate": self.drop_rate,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        drop=0,
        attn_drop=0,
        drop_path_rate=0,
        sr_ratio=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path_rate = drop_path_rate
        self.sr_ratio = sr_ratio

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
        self.attn = Attention(
            dim,
            num_heads,
            sr_ratio,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop,
            proj_drop_rate=drop,
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else tf.keras.layers.Layer()
        )
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop_rate=drop,
        )

    def call(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path_rate": self.drop_path_rate,
                "sr_ratio": self.sr_ratio,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class OverlapPatchEmbed(tf.keras.layers.Layer):
    def __init__(
        self, patch_size=7, stride=4, filters=768, **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.filters = filters

        self.pad = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=patch_size,
            strides=stride,
            padding="VALID",
            name='proj',
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-05)

    def call(self, x):
        x = self.conv(self.pad(x))
        get_shapes = tf.shape(x)
        H = get_shapes[1]
        W = get_shapes[2]
        C = get_shapes[3]
        x = tf.reshape(x, (-1, H * W, C))
        x = self.norm(x)
        return x, H, W
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "stride": self.stride,
                "filters": self.filters,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class MixVisionTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.qkv_bias = qkv_bias
        self.depths = depths
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.sr_ratios = sr_ratios

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            patch_size=7,
            stride=4,
            filters=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            filters=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            filters=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            filters=embed_dims[3],
        )

        dpr = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = [
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                sr_ratio=sr_ratios[0],
            )
            for i in range(depths[0])
        ]
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

        cur += depths[0]
        self.block2 = [
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                sr_ratio=sr_ratios[1],
            )
            for i in range(depths[1])
        ]
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

        cur += depths[1]
        self.block3 = [
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                sr_ratio=sr_ratios[2],
            )
            for i in range(depths[2])
        ]
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

        cur += depths[2]
        self.block4 = [
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                sr_ratio=sr_ratios[3],
            )
            for i in range(depths[3])
        ]
        self.norm4 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

    def call_features(self, x):
        B = tf.shape(x)[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, tf.shape(x)[-1]))
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = tf.reshape(x, (B, H, W, tf.shape(x)[-1]))
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = tf.reshape(x, (B, H, W, tf.shape(x)[-1]))
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = tf.reshape(x, (B, H, W, tf.shape(x)[-1]))
        outs.append(x)

        return outs

    def call(self, x):
        x = self.call_features(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dims": self.embed_dims,
                "num_heads": self.num_heads,
                "mlp_ratios": self.mlp_ratios,
                "qkv_bias": self.qkv_bias,
                "depths": self.depths,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "sr_ratios": self.sr_ratios,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs):
        resized = tf.image.resize(
            inputs,
            size=(self.height, self.width),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        return resized
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "drop_path": self.drop_path,
            }
        )
        return config

MODEL_CONFIGS = {
    "mit_b0": {
        "embed_dims": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_dim": 256,
    },
    "mit_b1": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decode_dim": 256,
    },
    "mit_b2": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decode_dim": 768,
    },
    "mit_b3": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decode_dim": 768,
    },
    "mit_b4": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "decode_dim": 768,
    },
    "mit_b5": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "decode_dim": 768,
    },
}

def SegFormer_B0(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = MixVisionTransformer(
        img_size=input_shape[1],
        embed_dims=MODEL_CONFIGS["mit_b0"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b0"]["depths"],
    )(input_layer)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b0"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.layers.Softmax(name='segmentation_output')(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)


def SegFormer_B1(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = MixVisionTransformer(
        embed_dims=MODEL_CONFIGS["mit_b1"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b1"]["depths"],
    )(input_layer)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b1"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.layers.Softmax(name='segmentation_output')(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)


def SegFormer_B2(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = MixVisionTransformer(
        embed_dims=MODEL_CONFIGS["mit_b2"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b2"]["depths"],
    )(input_layer)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b2"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.layers.Softmax(name='segmentation_output')(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)


def SegFormer_B3(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = MixVisionTransformer(
        embed_dims=MODEL_CONFIGS["mit_b3"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b3"]["depths"],
    )(input_layer)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b3"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.layers.Softmax(name='segmentation_output')(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)


def SegFormer_B4(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = MixVisionTransformer(
        embed_dims=MODEL_CONFIGS["mit_b4"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b4"]["depths"],
    )(input_layer)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b4"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.layers.Softmax(name='segmentation_output')(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)


def SegFormer_B5(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = MixVisionTransformer(
        embed_dims=MODEL_CONFIGS["mit_b5"]["embed_dims"],
        depths=MODEL_CONFIGS["mit_b5"]["depths"],
    )(input_layer)
    x = SegFormerHead(
        num_classes=num_classes,
        decode_dim=MODEL_CONFIGS["mit_b5"]["decode_dim"],
    )(x)

    x = ResizeLayer(input_shape[0], input_shape[1])(x)
    x = tf.keras.layers.Softmax(name='segmentation_output')(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)