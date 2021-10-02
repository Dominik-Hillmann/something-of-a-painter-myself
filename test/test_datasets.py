# Testing framework
import unittest2

# Things to be tested
from data.build import MonetDataset, PhotoDataset
from torch import Tensor


class DatasetTests(unittest2.TestCase):
    """Tests whether the Monet dataset correctly returns images."""

    def setUp(self) -> None:
        super().setUp()
        self.monet_dataset = MonetDataset()
        self.photo_dataset = PhotoDataset()


    def test_get_item(self) -> None:
        x_monet = self.monet_dataset[0]
        x_photo = self.monet_dataset[0]

        self.assertTrue(type(x_monet) is Tensor)
        self.assertTrue(type(x_photo) is Tensor)

    
    def test_get_len(self) -> None:
        self.assertTrue(len(self.monet_dataset) == 300)
        self.assertTrue(len(self.photo_dataset) == 7038)
    

    def test_out_of_bound_behavior(self) -> None:
        with self.assertRaises(IndexError):
            self.monet_dataset[300]
            self.photo_dataset[7038]

    
    def test_all_imgs_loaded_correctly(self) -> None:
        for i in range(len(self.monet_dataset)):
            loaded_img = self.monet_dataset[i]
            self.assertTrue(type(loaded_img) is Tensor)

            self.assertEqual(loaded_img.shape, (256, 256, 3))


        for i in range(len(self.photo_dataset)):
            loaded_img = self.photo_dataset[i]
            self.assertTrue(type(loaded_img) is Tensor)

            self.assertEqual(loaded_img.shape, (256, 256, 3))