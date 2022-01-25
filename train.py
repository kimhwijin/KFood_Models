import kfood_dataset
import kfood_model

def get_dataset():
    filepaths = kfood_dataset.get_image_paths()
    dataset = kfood_dataset.make_kfood_dataset(filepaths, batch_size=32)
    return dataset


dataset = get_dataset()
inception = kfood_model.make_inception()
inception.summary()
inception.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = inception.fit(dataset, epochs=1)