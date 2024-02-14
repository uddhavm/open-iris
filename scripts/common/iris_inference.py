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
# Options for the `eye_side` argument are: ["left", "right"]
output = iris_pipeline(img_data=img_pixels, eye_side="right")
output_2 = iris_pipeline(img_data=img_pixels_3, eye_side="right")
# wget https://wld-ml-ai-data-public.s3.amazonaws.com/public-iris-images/example_orb_image_1.png -O ./sample_ir_image.png

# print(output["iris_template"]["mask_codes"].shape)
template_probe = output["iris_template"]#["iris_codes"]

iris_pipeline.call_trace.get("encoder")

template_gallery = output_2 #["iris_template"]#["iris_codes"] 

print(output["metadata"])

# my_irisCode = np.array(my_irisCode)
# my_irisCode2 = np.array(my_irisCode2)

# array_bytes = my_irisCode.tobytes()
# np.sum(my_irisCode)

# print(np.sum(my_irisCode))

### Todo: use https://github.com/worldcoin/open-iris/blob/dev/src/iris/nodes/matcher/hamming_distance_matcher.py
# hamming_distance = np.mean(my_flat_iris_code!= my_flat_iris_code2)

# Assuming template_probe and template_gallery should be instances of a class with 'iris_codes' attribute

if 'iris_codes' in template_gallery:
    print("Yes queen")
# Check if template_probe is an instance of a class with 'iris_codes' attribute
if isinstance(template_probe, IrisTemplate) and hasattr(template_probe, 'iris_codes'):
    print("template_probe is of the correct class with 'iris_codes' attribute")
else:
    print("template_probe is not of the correct class or does not have 'iris_codes' attribute")

# Check if template_gallery is an instance of a class with 'iris_codes' attribute
if isinstance(template_gallery, HammingDistanceMatcher) and hasattr(template_gallery, 'iris_codes'):
    print("template_gallery is of the correct class with 'iris_codes' attribute")
else:
    print("template_gallery is not of the correct class or does not have 'iris_codes' attribute")


# template_probe = my_irisCode  # Your probe template initialization
# template_gallery = my_irisCode2  # Your gallery template initialization
# Create an instance of HammingDistanceMatcher
hamming_distance_matcher = HammingDistanceMatcher(rotation_shift=15, nm_dist=None, weights=None)
matching_distance = hamming_distance_matcher.run(template_probe['iris_codes'], template_gallery['iris_codes'])

print(f"Matching Distance: {matching_distance}")

### for hashing only 
# hash_value = hash(array_bytes)

# print(my_flat_iris_code.shape)
# print(hash_value)

canvas = iris_visualizer.plot_ir_image(iris.IRImage(img_data=img_pixels, eye_side="right"))
plt.show()

canvas = iris_visualizer.plot_iris_template(output["iris_template"])
plt.show();



# output["error"] is None
# output.keys()


# print(output["metadata"])
# print("""`output["iris_template"]` value types are: """ + type(output["iris_template"]["iris_codes"]).__name__ + ", " + type(output["iris_template"]["mask_codes"]).__name__)
# print(output["metadata"])
# /Users/uddhav/open-iris