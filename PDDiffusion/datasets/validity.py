from PIL import Image
import os.path, PIL

def image_is_valid(file):
    """Test if an image file on disk loads with PIL."""

    try:
        test_image = Image.open(os.path.abspath(file))
        try:
            test_image.load()
            test_image.close()

            return True
        except PIL.UnidentifiedImageError:
            print ("Warning: Image {} is an unknown format".format(file))
            test_image.close()

            return False
    except PIL.Image.DecompressionBombError:
        print ("Warning: Image {} is too large for PIL".format(file))
        return False
    except OSError as e:
        print ("Warning: Image {} could not be read from disk, error was: {}".format(file, e))
        return False