import tensorflow as tf
if 1:
    print("abc")
print(tf.test.is_built_with_cuda())
#tf.test.is_gpu_available(cuda_only=True)
print(tf.config.list_physical_devices('GPU'))
print(list(range(3)))
print(3%4)
