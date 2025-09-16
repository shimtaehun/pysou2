import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU 감지됨:", gpus)
else:
    print("❌ GPU 없음, CPU 사용 중")
