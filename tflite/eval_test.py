import numpy as np
import tflite_runtime.interpreter as tflite
import random
import time
from contextlib import contextmanager


@contextmanager
def timer():
    st_time = time.time()
    yield
    print("Total time %.3f seconds" % (time.time() - st_time))


not_optimized_model_path = "data/model.tflite_NoOptimize"
optimized_model_path = "data/model.tflite_NoOptimize"

data = np.load("data/data.npy")
target = np.load("data/target.npy")

DATA_SIZE = data.shape[0]

NUM_TESTS = 5000
print("Predictions count for each model:", NUM_TESTS)

tflite_optimized_model_interpreter = tflite.Interpreter(model_path=optimized_model_path)
tflite_not_optimized_model_interpreter = tflite.Interpreter(model_path=not_optimized_model_path)
'''
You can change input shape for batch support before allocate_tensors(). But it needs fixed size
BATCH_SIZE = 2
tflite_not_optimized_model_interpreter.resize_tensor_input(tflite_not_optimized_model_interpreter.get_input_details()[0]["index"], [BATCH_SIZE, 28, 28,  1])
'''
tflite_optimized_model_interpreter.allocate_tensors()
tflite_not_optimized_model_interpreter.allocate_tensors()

models = {}
models.update({"Not quantized TFLite model": tflite_not_optimized_model_interpreter})
models.update({"Quantized TFLite model": tflite_optimized_model_interpreter})

# print(tflite_not_optimized_model_interpreter.get_input_details())
input_X_index = tflite_not_optimized_model_interpreter.get_input_details()[0]["index"]
# print("input_X_index:", input_X_index)

# print(tflite_not_optimized_model_interpreter.get_output_details())
output_y_index = tflite_not_optimized_model_interpreter.get_output_details()[0]["index"]
# print("output_y_index:", output_y_index)


for model_name in models:
    print()
    print(model_name + ":", flush=True)
    with timer():
        for iii in range(NUM_TESTS):
            tmp_idx = random.randint(0, DATA_SIZE - 1)
            tmp_data = data[tmp_idx:tmp_idx + 1]
            tmp_target = target[tmp_idx:tmp_idx + 1]
            models[model_name].set_tensor(input_X_index, tmp_data)
            models[model_name].invoke()
        output = models[model_name].tensor(output_y_index)
        # print("Target: %d; Prediction: %d" % (tmp_target[0], np.argmax(output()[0])))
