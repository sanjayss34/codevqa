"""Write Python code to answer the questions about each image."""
# Global constants
# min x coordinate
LEFT = 0
# min y coordinate
BOTTOM = 0
# max x coordinate
RIGHT = 24
# max y coordinate
TOP = 24
from PIL import Image
from utils import open_image, query, get_pos

"""
API Reference:
open_image(path: str) -> Image - opens the image at the path and returns it as an Image object
query(img: Image, question: str) -> str - queries the image returns an answer to the question
get_pos(img: Image, object: str) -> (float, float) - returns the position of the object in the image 
"""
