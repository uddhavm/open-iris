import cv2
import iris
import matplotlib.pyplot as plt
import numpy as np
from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher

# 1. Create IRISPipeline object
iris_pipeline = iris.IRISPipeline()
iris_visualizer = iris.visualisation.IRISVisualizer()

# 2. Load IR image of an eye
# img_pixels = cv2.imread("https://wld-ml-ai-data-public.s3.amazonaws.com/public-iris-images/example_orb_image_1.png", cv2.IMREAD_GRAYSCALE)
img_pixels = cv2.imread("/Users/uddhav/open-iris/sample_ir_image.png", cv2.IMREAD_GRAYSCALE)
img_pixels_3 = cv2.imread("/Users/uddhav/open-iris/sample_ir_image_3.png", cv2.IMREAD_GRAYSCALE)

# 3. Perform inference
output = iris_pipeline(img_data=img_pixels, eye_side="right")
first_iris_template = iris_pipeline.call_trace.get("encoder")

# get intermediate result of every node that builds iris pipeline
output_2 = iris_pipeline(img_data=img_pixels_3, eye_side="right")
second_iris_template = iris_pipeline.call_trace.get("encoder")

print(f"Type of first {type(first_iris_template)}.\tType of first {type(second_iris_template)}")

# print(output["iris_template"]["mask_codes"].shape)
# iris_pipeline.call_trace.get("encoder").iris_codes
# corresponds to 2 gabor filters (numpy arrays) for low and high freq of iris textures

# Create an instance of HammingDistanceMatcher
hamming_distance_matcher = HammingDistanceMatcher(rotation_shift=180, nm_dist=None, weights=None)
# rotation_shift = tilt of head, rotating bits by moving +/- along phi (x) and compare 1 to 2nd-rotated, output min 
matching_distance = hamming_distance_matcher.run(first_iris_template, second_iris_template)
print(f"Matching Distance: {matching_distance}")

canvas = iris_visualizer.plot_ir_image(iris.IRImage(img_data=img_pixels, eye_side="right"))
plt.show()

canvas = iris_visualizer.plot_iris_template(output["iris_template"])
plt.show()