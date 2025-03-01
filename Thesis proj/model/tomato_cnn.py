import tensorflow as tf
import os

def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',        # Using pre-trained weights
        include_top=False,         # Removing classification layers
        input_shape=(96, 96, 3),  # Small input size for ESP32-CAM
        alpha=0.35                 # Makes network 65% smaller
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),  # Smaller dense layer
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def convert_to_tflite(model, filename):
    # Optimize for ESP32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_dataset_gen
    
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)

def representative_dataset_gen():
    # Generate calibration data for quantization
    dataset_path = "../data/train"
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path)[:100]:
            img_path = os.path.join(class_path, img_name)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(96, 96)
            )
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = x / 255.0
            x = tf.expand_dims(x, axis=0)
            yield [x]

def prepare_dataset(data_dir, img_size=(96, 96), batch_size=32):
    # Use tf.keras.utils instead of keras.preprocessing.image
    train_datagen = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    valid_datagen = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    # Normalize the data
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_datagen = train_datagen.map(lambda x, y: (normalization_layer(x), y))
    valid_datagen = valid_datagen.map(lambda x, y: (normalization_layer(x), y))
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    
    train_datagen = train_datagen.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )
    
    return train_datagen, valid_datagen

# Two-phase training
if __name__ == "__main__":
    DATA_DIR = '../data'
    MODEL_PATH = '../esp32/model/tomato_model.tflite'
    
    # Prepare dataset
    train_generator, valid_generator = prepare_dataset(DATA_DIR)
    num_classes = len(train_generator.class_indices)
    
    # Create and train model
    model = create_model(num_classes)
    
    # Phase 1: Train only top layers
    model.fit(train_generator, epochs=10, validation_data=valid_generator)
    
    # Phase 2: Fine-tuning - unfreeze some layers
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Only unfreeze the last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune
    history_fine = model.fit(
        train_generator,
        epochs=5,
        validation_data=valid_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    # Convert and save model
    convert_to_tflite(model, MODEL_PATH)
