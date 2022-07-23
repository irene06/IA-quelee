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
#definimos la estructura de la red, especificando la cantidad de capas ocultas y densas y con 64 neuronas cada una
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),#para las capas ocultas usamos relu
    tf.keras.layers.Flatten(10,activation=tf.nn.softmax)#para las capas de salida
    
])
#compilamos el modelo e indicamos las funciones que vamos a usar
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
#Aprendizaje va a ser por lotes de 32 cada lote
BATCHSIZE= 32
train_dataset=train_dataset.repeat().shuffle(num_train_examples)
test_dataset = test_dataset.batch(BATCHSIZE)
#Realizando el aprendizaje
model.fit(
    train_dataset, epochs=5,
    steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE)
)
#Evaluamos nuestro modelo ya entrenado contra el dataset de pruebas
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32)
)

print("Resultado en las pruebas:", test_accuracy)

for test_images, test_labels in dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i],images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[...,0], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color='blue'
    else:
        color='red'

    plt.xlabel("Prediccion:{}".format(class_names[predicted_label]), color=color)
