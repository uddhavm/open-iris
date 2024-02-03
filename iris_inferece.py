import cv2
import iris
import matplotlib.pyplot as plt
import numpy as np
import hashlib

# 1. Create IRISPipeline object
iris_pipeline = iris.IRISPipeline()
iris_visualizer = iris.visualisation.IRISVisualizer()
# 2. Load IR image of an eye

# img_pixels = cv2.imread("https://wld-ml-ai-data-public.s3.amazonaws.com/public-iris-images/example_orb_image_1.png", cv2.IMREAD_GRAYSCALE)

img_pixels = cv2.imread("/Users/uddhav/open-iris/sample_ir_image.png", cv2.IMREAD_GRAYSCALE)
img_pixels_3 = cv2.imread("/Users/uddhav/open-iris/sample_ir_image_3.png", cv2.IMREAD_GRAYSCALE)

# 3. Perform inference
# Options for the `eye_side` argument are: ["left", "right"]
output = iris_pipeline(img_data=img_pixels, eye_side="right")
output_2 = iris_pipeline(img_data=img_pixels_3, eye_side="right")
# wget https://wld-ml-ai-data-public.s3.amazonaws.com/public-iris-images/example_orb_image_1.png -O ./sample_ir_image.png

# print(output["iris_template"]["mask_codes"].shape)
my_irisCode = output["iris_template"]["iris_codes"]
my_irisCode2 = output_2["iris_template"]["iris_codes"] 

print(output_2["metadata"])

my_irisCode = np.array(my_irisCode)
my_irisCode2 = np.array(my_irisCode2)

array_bytes = my_irisCode.tobytes()
# np.sum(my_irisCode)
my_flat_iris_code = my_irisCode.flatten()
my_flat_iris_code2 = my_irisCode2.flatten()

print(np.sum(my_irisCode))

hamming_distance = np.mean(my_flat_iris_code!= my_flat_iris_code2)

print(hamming_distance)
### for hashing only 
# hash_value = hash(array_bytes)

# print(my_flat_iris_code.shape)
# print(hash_value)

# canvas = iris_visualizer.plot_ir_image(iris.IRImage(img_data=img_pixels, eye_side="right"))
# plt.show()

# canvas = iris_visualizer.plot_iris_template(output["iris_template"])
# plt.show();



# output["error"] is None
# output.keys()


# print(output["metadata"])
# print("""`output["iris_template"]` value types are: """ + type(output["iris_template"]["iris_codes"]).__name__ + ", " + type(output["iris_template"]["mask_codes"]).__name__)
# print(output["metadata"])
# /Users/uddhav/open-iris