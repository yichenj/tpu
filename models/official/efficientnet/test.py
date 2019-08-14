import tensorflow as tf

def parse_fn(filename):
  print('Parse filename %s' % filename)
  img = tf.io.read_file(filename)
  img = tf.io.decode_png(img, channels=3)
  img = tf.image.resize(img, [224, 224])
  img /= 255.0

  label = tf.strings.split(filename, '/')
  out = table.lookup(label[-2])
  return img, out


table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(["normal", "dead", "rotten", "clear"], [0, 1, 2, 3], key_dtype=tf.string, value_dtype=tf.int64), 4)
dataset = tf.data.Dataset.list_files('/mnt/projects/Data/Training/*/*.png')
dataset = dataset.shuffle(True)
dataset = dataset.repeat()
dataset = dataset.map(map_func=parse_fn)
dataset = dataset.batch(2)

for x in dataset:
  print(x)
