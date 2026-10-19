import io
import unittest

import numpy as np
from PIL import Image

from serve.deepseek41_images import IMAGE_TOKEN, ImagePatches, expand_images, image_grid, preprocess, round_bf16


class ImageTests(unittest.TestCase):
    def test_extreme_grids(self):
        for w, h in [(1, 1), (43, 71), (4096, 8192), (1, 20000), (20000, 1), (777, 1023)]:
            lh, lw, ph, pw = image_grid(w, h)
            self.assertEqual((ph % 14, pw % 14), (0, 0))
            self.assertLessEqual(lh * (lw + 1) + 2, 1024)
            self.assertLessEqual((ph // 14) * (pw // 14), 9198)
        for shape in [(0, 1), (1, 0), (-1, 1)]:
            with self.assertRaises(ValueError):
                image_grid(*shape)

    def test_pixels_and_exif(self):
        image = Image.new("RGB", (14, 28), (255, 0, 127))
        image.getexif()[274] = 6
        data = io.BytesIO()
        image.save(data, format="PNG", exif=image.getexif())
        p = preprocess(data.getvalue())
        self.assertGreater(p.height, p.width)
        self.assertEqual(p.values.shape, (p.height * p.width, 3, 14, 14))
        self.assertEqual(p.values.dtype, np.float32)
        # Source pixels remain vertical, and normalization is rounded exactly to BF16.
        self.assertGreater(np.mean(p.values[:, 0] == 1), 0.95)
        self.assertGreater(np.mean(p.values[:, 1] == -1), 0.95)
        np.testing.assert_array_equal(p.values[len(p.values) // 2, 0], np.ones((14, 14), np.float32))
        self.assertTrue(np.all((p.values.view(np.uint32) & 0xffff) == 0))
        with self.assertRaises(ValueError):
            preprocess(b"not an image")

    def test_round_even(self):
        x = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], dtype=np.uint32).view(np.float32)
        np.testing.assert_array_equal(round_bf16(x).view(np.uint32), [0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000])

    def test_two_images_keep_whole_spans(self):
        p = ImagePatches(np.zeros((9, 3, 14, 14), np.float32), 3, 3, 1, 1)
        ids, spans = expand_images([7, IMAGE_TOKEN, 8, IMAGE_TOKEN, 9], [p, p])
        self.assertEqual(ids, [7] + [IMAGE_TOKEN] * 4 + [8] + [IMAGE_TOKEN] * 4 + [9])
        self.assertEqual([x[0] for x in spans], [1, 6])
        with self.assertRaises(ValueError):
            expand_images([7], [p])


if __name__ == "__main__":
    unittest.main()
