#IA who reads
logger = tf.get_logger()

logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load('mnist',as_supervised=True,with_info=True)
train_dataset, test_dataset=dataset['train'],dataset['test']

class_names =[
    'Zero','One','Two','Three','Four','Five','Six',
    'Seven', 'Eight','Nine'
]
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#Normalize Numbers from 0 to 255
def normalize(images,labels)
    images = tf.cast(images,tf.float32)
    images/= 255
    return images, labels
train_dataset = train_dataset.map(normalize)
test_dataset= test_dataset.map(normalize)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=())
])