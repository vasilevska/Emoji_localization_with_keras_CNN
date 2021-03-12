import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from PIL import Image, ImageDraw
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

def create_example():
  class_id = np.random.randint(0,9)
  scale = np.random.randint(5,10)/10
  size = math.floor(72*scale)
  row = np.random.randint(0,144-size)
  col = np.random.randint(0,144-size)

  image = Image.new("RGB", (144,144), (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))
  second = (emojis[class_id]['image']).resize((size, size)).rotate(15*np.random.randint(0,23))
  image.paste(second, (col, row), second.convert('RGBA'))
  image = np.array(image)
  
  return image.astype('uint8'), class_id, (row+10*scale)/144, (col+10*scale)/144, math.floor(size - 2*10*scale)

def plot_bounding_box(image, size, gt_coords, pred_size = 0, pred_coords=[], norm = False):
  if norm:
    image *= 255.
    image = image.astype('uint8')
  
  image = Image.fromarray(image)
  draw = ImageDraw.Draw(image)

  row, col = gt_coords
  row *= 144
  col *= 144
  draw.rectangle((col, row, col+size, row+size), outline = 'green', width = 3)
  
  if len(pred_coords) == 2:
    row, col = pred_coords
    row *= 144
    col *= 144
    draw.rectangle((col, row, col+pred_size, row+pred_size), outline = 'red', width = 3)
  return image


def data_generator(batch_size = 16):
  while True:
    x_batch = np.zeros((batch_size, 144, 144, 3))
    y_batch = np.zeros((batch_size, 9))
    bbox_batch = np.zeros((batch_size, 3))

    for i in range(0, batch_size):
      image, class_id, row, col,size = create_example()
      x_batch[i] = image/255.
      y_batch[i, class_id] = 1.0
      bbox_batch[i] = np.array([row,col,size])
    yield {'image':x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}


class IoU(tf.keras.metrics.Metric):
  def __init__ (self, **kwargs):
    super(IoU, self).__init__(**kwargs)

    self.iou = self.add_weight(name = 'iou', initializer = 'zeros')
    self.total_iou = self.add_weight(name = 'total_iou', initializer = 'zeros')
    self.num_ex = self.add_weight(name = 'num_ex', initializer = 'zeros')
  
  def update_state(self, y_true, y_pred, sample_weight = None):
    def get_box(y):
      rows, cols, size = y[:, 0], y[:,1], y[:,2]
      rows, cols = rows * 144, cols * 144
      y1, y2 = rows, rows + size*52
      x1, x2 = cols, cols + size*52
      return x1, y1, x2, y2, size
    
    def get_area(x1, y1, x2, y2):
      return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)

    gt_x1, gt_y1, gt_x2, gt_y2, gsize = get_box(y_true)
    p_x1, p_y1, p_x2, p_y2, psize = get_box(y_pred)

    # gt_x1 *= 72/gsize
    # gt_y1 *= 72/gsize
    # gt_x2 *= 72/gsize
    # gt_y2 *= 72/gsize

    # p_x1  *= 72/psize
    # p_y1  *= 72/psize 
    # p_x2  *= 72/psize 
    # p_y2  *= 72/psize

    i_x1 = tf.maximum(gt_x1, p_x1)
    i_y1 = tf.maximum(gt_y1, p_y1)
    i_x2 = tf.maximum(gt_x2, p_x2)
    i_y2 = tf.maximum(gt_y2, p_y2)

    i_area = get_area(i_x1, i_y1, i_x2, i_y2)
    u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area

    iou = tf.math.divide(i_area, u_area)
    self.num_ex.assign_add(1)
    self.total_iou.assign_add(tf.reduce_mean(iou))
    self.iou = tf.math.divide(self.total_iou, self.num_ex)

  def result(self):
    return self.iou

  def reset_state(self):
    self.iou = self.add_weight(name = 'iou', initializer = 'zeros')
    self.total_iou = self.add_weight(name = 'total_iou', initializer = 'zeros')
    self.num_ex = self.add_weight(name = 'num_ex', initializer = 'zeros')

def test_model(model, test_datagen):
  example, label = next(test_datagen)
  x = example['image']
  y = label['class_out']
  box = label['box_out']

  pred_y, pred_box = model.predict(x)

  px,py, psize = pred_box[0]
  gx, gy, gsize = box[0]
  pred_coords = [px,py]
  gt_coords = [gx,gy]
  pred_class = np.argmax(pred_y[0])
  image = x[0]
  
  gt = emojis[np.argmax(y[0])]['name']
  pred_class_name = emojis[pred_class]['name']

  image = plot_bounding_box(image, gsize, gt_coords, psize, pred_coords, norm=True)
  color = 'green' if gt == pred_class_name else 'red'

  plt.imshow(image)
  plt.xlabel(f'Pred: {pred_class_name}', color = color)
  plt.ylabel(f'GT: {gt}', color = color)
  plt.xticks([])
  plt.yticks([])

def test(model):
  test_datagen = data_generator(1)
  plt.figure(figsize=(16,4))
  for i in range(0,6):
    plt.subplot(1,6, i + 1)
    test_model(model, test_datagen)
  plt.show()

class ShowTestImages(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = None):
    test(self.model)


def lr_schedule(epoch, lr):
  if(epoch + 1) % 5 == 0:
    lr *= 0.2
  return max(lr, 3e-7)



emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
    emojis[class_id]['image'] = png_file

image, class_id, row, col, size = create_example()
plt.imshow(image)

image = plot_bounding_box(image, size, gt_coords = [row, col])
plt.imshow(image)
plt.title(emojis[class_id]['name'])
plt.show()

example, label = next(data_generator(1))
image = example['image'][0]
class_id = np.argmax(label['class_out'][0])
x,y,size = label['box_out'][0]
coords = [x,y]
image = plot_bounding_box(image, size, coords, norm = True)
plt.imshow(image)
plt.title(emojis[class_id]['name'])
plt.show()


input_ = Input(shape=(144,144,3), name = 'image') #ime odgovara imenu iz generatora
x = input_
for i in range(0,6):
  n_filters = 2**(4 + i)
  x = Conv2D(n_filters, 3, activation = 'relu', padding="same")(x)
  x = BatchNormalization()(x)
  x = MaxPool2D(2)(x)

x = Flatten()(x)
x = Dense(256, activation = 'relu')(x)

class_out = Dense(9, activation='softmax', name = 'class_out')(x)
box_out = Dense(3, activation = 'linear', name = 'box_out')(x)

model = tf.keras.models.Model(input_, [class_out, box_out])
#model.summary()

model.compile(
    loss = {
        'class_out' : 'categorical_crossentropy',
        'box_out' : 'mse'
    },
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    metrics = {
        'class_out' : 'accuracy',
        'box_out' : IoU(name = 'iou')
    }
)
test(model)

_= model.fit(
    data_generator(),
    epochs = 100,
    steps_per_epoch = 500,
    callbacks = [
                 ShowTestImages(),
                 tf.keras.callbacks.EarlyStopping(monitor = 'box_out_iou', patience = 100, mode = 'max'),
                 tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    ]
)
