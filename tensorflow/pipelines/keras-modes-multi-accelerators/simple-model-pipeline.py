from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset
from typing import NamedTuple


project_id = ''

pipeline_root_path = 'gs://pipeline-tester3'


@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets', 'numpy==1.21.6']
)
def ingest_data() -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    print("\n\n" + "Tensorflow version is: " + tf.__version__ + "\n\n")
    print("\n\n" + "Tfds version is: " + tfds.__version__ + "\n\n")

    validation_split = 10
    bucket = 'gs://pipeline-tester3'

    test_ds = tfds.load('cifar10', split='test', as_supervised=True, shuffle_files=True)
    validation_ds = tfds.load('cifar10', split=f'train[:{validation_split}%]', as_supervised=True)
    training_ds = tfds.load('cifar10', split=f'train[{validation_split}%:]', as_supervised=True)

    def custom_shard_func(self, element):
        return np.int64(0)

    training_ds.save(
        path=bucket + "/train_ds", shard_func=custom_shard_func)

    validation_ds.save(
        path=bucket + "/valid_ds", shard_func=custom_shard_func)

    test_ds.save(
        path=bucket + "/test_ds", shard_func=custom_shard_func)

    return bucket


@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'keras'],
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.2-11',
)
def create_model(text: str) -> str:
    import tensorflow as tf
    from keras import applications

    bucket = 'gs://pipeline-tester3'

    # check for GPU:
    print('\n\n GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print('\n\n')

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        new_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer((32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('\n\n' + str(new_model.summary()) + '\n\n')
    new_model.save(bucket + "/untrained-model")
    return "model saved:" + bucket


@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets'],
    output_component_file="train_model.yaml",
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.2-11',
)
def train_model(text: str) -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf
    import os
    import datetime

    # check for GPU:
    print('\n\n GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print('\n\n')

    #Storage buckets
    # bucket = 'gs://tfds-dir3'
    bucket = 'gs://pipeline-tester3'

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    # batch size
    batch_size = 32 * strategy.num_replicas_in_sync

    # load data from gcs bucket
    model_name = 'simple model'
    train_data = tf.data.Dataset.load(bucket + "/train_ds")
    train_data = train_data.map(lambda f, l: (tf.cast(f, tf.float64) / 255, l))
    train_data = train_data.shuffle(buffer_size=5000)
    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("\n\n finish loading training data \n\n")

    valid_data = tf.data.Dataset.load(bucket + "/valid_ds")
    valid_data = valid_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("\n\n finish loading validation data \n\n")
    test_data = tf.data.Dataset.load(bucket + "/test_ds")
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("\n\n finish loading test data \n\n")

    # load model from gcs
    model = tf.keras.models.load_model(bucket + '/untrained-model')

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=bucket + f'/ckpts/cifar10-{model_name}-' + '{epoch:02d}-{val_accuracy:.4f}')
    log_dir = bucket + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=20)
    print('\n\n history\n' + str(history) + '\n\n')

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print('\n\n'  + f'Test accuracy: {test_acc * 100:.2f}%' + '\n\n')

    #Save the model
    model.save(bucket + "/simple_ model")

    return "model trained"



@dsl.pipeline(
    name='simple-pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def simple_pipeline():
    ingestion_task = ingest_data()
    create_model_task = create_model(text=ingestion_task.output).set_accelerator_type('NVIDIA_TESLA_V100').set_cpu_limit('4').set_memory_limit('16G').set_accelerator_limit(2)
    train_model_task_4_V100_GPUs = train_model(text=create_model_task.output)\
        .set_accelerator_type('NVIDIA_TESLA_V100')\
        .set_cpu_limit('4')\
        .set_memory_limit('16G')\
        .set_accelerator_limit(2)\
        .set_display_name('2 x V100 GPUS ')
    train_model_task_2_V100_GPUs = train_model(text=create_model_task.output)\
        .set_accelerator_type('NVIDIA_TESLA_V100')\
        .set_cpu_limit('4')\
        .set_memory_limit('16G')\
        .set_accelerator_limit(1)\
        .set_display_name('1 x V100 GPUS ')


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=simple_pipeline,
        package_path='simple-model-pipeline.json'
    )
