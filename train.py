import kfood_dataset

def get_dataset():
    filepaths = kfood_dataset.get_image_paths()
    dataset = kfood_dataset.make_kfood_dataset(filepaths, batch_size=32)
    return dataset