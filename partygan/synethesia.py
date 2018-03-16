def random_start_img(batch_size, size=(1024, 512)):
    generation_shape = (batch_size, *size, 3)
    img = tf.random_uniform(shape=generation_shape, minval=0., maxval=1.,
                            dtype=tf.float32, name="random_start_img")
    return img
