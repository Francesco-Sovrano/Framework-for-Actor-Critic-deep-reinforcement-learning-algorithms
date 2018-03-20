
import h5py
import json
import numpy as np
import keras.backend as K
from keras import optimizers
from keras.models import Sequential


def save_optmizer_to_hdf5_file(model, id, hdf5_file):
    """
    :type model: keras.models.Model
    :type id: str
    :type hdf5_file: h5py.File
    """
    model_group = hdf5_file.create_group(id)
    save_optmizer_to_hdf5_group(model, model_group)


def save_optmizer_to_hdf5_group(model, hdf5_group):
    """
    :type model: keras.models.Model
    :param hdf5_group: h5py.Group
    """
    hdf5_group.attrs['training_config'] = json.dumps({
        'optimizer_config': {
            'class_name': model.optimizer.__class__.__name__,
            'config': model.optimizer.get_config()
        },
        'loss': model.loss,
        'metrics': model.metrics,
        'sample_weight_mode': model.sample_weight_mode,
        'loss_weights': model.loss_weights,
    }, default=get_json_type).encode('utf8')

    # Save optimizer weights.
    symbolic_weights = getattr(model.optimizer, 'weights')
    if symbolic_weights:
        optimizer_weights_group = hdf5_group.create_group('optimizer_weights')
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights,
                                         weight_values)):
            # Default values of symbolic_weights is /variable
            # for theano and cntk
            if K.backend() == 'theano' or K.backend() == 'cntk':
                if hasattr(w, 'name'):
                    if w.name.split('/')[-1] == 'variable':
                        name = str(w.name) + '_' + str(i)
                    else:
                        name = str(w.name)
                else:
                    name = 'param_' + str(i)
            else:
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))
        optimizer_weights_group.attrs['weight_names'] = weight_names
        for name, val in zip(weight_names, weight_values):
            param_dset = optimizer_weights_group.create_dataset(
                name,
                val.shape,
                dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val


def load_optmizer_from_hdf5_file(model, id, hdf5_file, custom_objects=None):
    """
    :type model: keras.models.Model
    :type id: str
    :type hdf5_file: h5py.File
    :param custom_objects: dict of non-keras objects that needs to be loaded
    """
    model_group = hdf5_file[id]
    load_optmizer_from_hdf5_group(model, model_group, custom_objects=custom_objects)


def load_optmizer_from_hdf5_group(model, hdf5_group, custom_objects=None):
    """
    :type model: keras.models.Model
    :type hdf5_group: h5py.Group
    """
    if custom_objects is None:
        custom_objects = {}

    # instantiate optimizer
    training_config = hdf5_group.attrs.get('training_config')
    training_config = json.loads(training_config.decode('utf-8'))
    optimizer_config = training_config['optimizer_config']
    optimizer = optimizers.deserialize(optimizer_config, custom_objects=custom_objects)

    # Recover loss functions and metrics.
    loss = convert_custom_objects(training_config['loss'], custom_objects)
    metrics = convert_custom_objects(training_config['metrics'], custom_objects)
    sample_weight_mode = training_config['sample_weight_mode']
    loss_weights = training_config['loss_weights']

    # Compile model.
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode)

    # Set optimizer weights.
    if 'optimizer_weights' in hdf5_group:
        # Build train function (to get weight updates).
        if isinstance(model, Sequential):
            model.model._make_train_function()
        else:
            model._make_train_function()
        optimizer_weights_group = hdf5_group['optimizer_weights']
        optimizer_weight_names = [n.decode('utf8') for n in
                                  optimizer_weights_group.attrs['weight_names']]
        optimizer_weight_values = [optimizer_weights_group[n] for n in
                                   optimizer_weight_names]
        model.optimizer.set_weights(optimizer_weight_values)


def get_json_type(obj):
    """ (From Keras library)
    Serialize any object to a JSON-serializable structure.

    # Arguments
        obj: the object to serialize

    # Returns
        JSON-serializable structure representing `obj`.

    # Raises
        TypeError: if `obj` cannot be serialized.
    """
    # if obj is a serializable Keras class instance
    # e.g. optimizer, layer
    if hasattr(obj, 'get_config'):
        return {'class_name': obj.__class__.__name__,
                'config': obj.get_config()}

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return {'type': type(obj),
                    'value': obj.tolist()}
        else:
            return obj.item()

    # misc functions (e.g. loss function)
    if callable(obj):
        return obj.__name__

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable:', obj)


def convert_custom_objects(obj, custom_objects):
    """ (From Keras library)
    Handles custom object lookup.

    # Arguments
        obj: object, dict, or list.
        custom_objects: dict of custom objects.

    # Returns
        The same structure, where occurrences
            of a custom object name have been replaced
            with the custom object.
    """
    if isinstance(obj, list):
        deserialized = []
        for value in obj:
            deserialized.append(convert_custom_objects(value, custom_objects))
        return deserialized
    if isinstance(obj, dict):
        deserialized = {}
        for key, value in obj.items():
            deserialized[key] = convert_custom_objects(value, custom_objects)
        return deserialized
    if obj in custom_objects:
        return custom_objects[obj]
    return obj