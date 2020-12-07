import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

parser = argparse.ArgumentParser(description="Convert model to tflite", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_folder", type=str, help="Input model folder path")
parser.add_argument("--output", type=str, help="Output tflite file")
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(args.model_folder)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.experimental_new_converter = True
tfmodel = converter.convert()
open(args.output, "wb").write(tfmodel)
