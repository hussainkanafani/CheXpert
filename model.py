import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.applications.densenet import DenseNet121


class ModelFactory:
    def __init__(self):
        self.models_ = dict(
            DenseNet121=dict(
                input_shape=(320, 320, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            ResNet50=dict(
                input_shape=(320, 320, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            )
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]

    def get_model(self, class_names, model_name="DenseNet121",
                  weights_path=None, input_shape=(320, 320, 3)):

        base_model_class = getattr(
            importlib.import_module(
                "keras.applications." + self.models_[model_name]['module_name']
            ),
            model_name)
        
        img_input = Input(shape=input_shape)        
        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights="imagenet",
            pooling="avg")
        x = base_model.output
        predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)

        if weights_path is not None:
            model.load_weights(weights_path)

        return model