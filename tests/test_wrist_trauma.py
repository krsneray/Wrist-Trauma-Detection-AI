import unittest
import torch
from src.utils.annotation_utils import flip_bounding_box_horizontal, remap_class
from src.utils.patient_utils import check_patient_leakage, get_age_group
from src.models.cbam_model import DummyAttentionLayer

class TestWristTraumaProject(unittest.TestCase):

    def test_flip_horizontal(self):
        self.assertEqual(flip_bounding_box_horizontal(100, 640), 540)
        with self.assertRaises(ValueError):
            flip_bounding_box_horizontal(700, 640)

    def test_remap_class(self):
        self.assertEqual(remap_class(4), 2)
        self.assertIsNone(remap_class(8))
        with self.assertRaises(KeyError):
            remap_class(99)

    def test_patient_leakage(self):
        self.assertTrue(check_patient_leakage(['P1', 'P2'], ['P3'], ['P4']))
        with self.assertRaises(ValueError):
            check_patient_leakage(['P1', 'P2'], ['P2', 'P3'], ['P4'])

    def test_age_grouping(self):
        self.assertEqual(get_age_group(12), '9-12')

    def test_attention_output_shape(self):
        dummy_input = torch.randn(4, 64, 128, 128)
        attention_layer = DummyAttentionLayer(channels=64)
        output = attention_layer(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

if __name__ == '__main__':
    unittest.main()     