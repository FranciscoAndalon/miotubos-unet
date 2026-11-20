"""Checks for label integrity, split leakage and segmentation behavior."""
import csv
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from myotubes import ROOT, SmallUNet, confusion, normalize, scores, threshold_search


class PipelineChecks(unittest.TestCase):
    def manifest(self):
        with (ROOT/'outputs/manifest.csv').open() as f:
            return list(csv.DictReader(f))

    def test_group_and_duplicate_isolation(self):
        rows = self.manifest()
        for key in ['file','group','sha256']:
            seen = {}
            for row in rows:
                self.assertEqual(seen.setdefault(row[key],row['split']),row['split'])
        self.assertEqual({r['split'] for r in rows},{'train','val','test'})
        self.assertEqual(len(rows),len(list((ROOT/'vt/images').glob('*.png'))))

    def test_mask_export_agreement(self):
        rows = self.manifest()
        # Allow tiny rounding differences between polygon export implementations.
        self.assertLess(max(float(r['coco_mask_disagreement']) for r in rows),.001)

    def test_metrics_with_foreground_and_empty_masks(self):
        y=np.array([1,1,0,0]);p=np.array([1,0,1,0])
        s=scores(confusion(p,y))
        self.assertAlmostEqual(s['dice'],.5)
        self.assertAlmostEqual(s['iou'],1/3)
        self.assertEqual(scores(confusion(y*0,y*0))['dice'],1)
        self.assertEqual(scores(confusion(y*0,y))['dice'],0)

    def test_network_accepts_original_aspect_and_odd_shapes(self):
        torch.set_num_threads(2)
        net=SmallUNet().eval()
        with torch.no_grad():
            for h,w in [(48,64),(49,65)]:
                prediction=net(torch.zeros(2,1,h,w))
                self.assertEqual(tuple(prediction.shape),(2,1,h,w))
                self.assertTrue(torch.isfinite(prediction).all())

    def test_constant_image_normalization(self):
        x=normalize(np.ones((10,10),dtype=np.uint16)*1000)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(((x>=0)&(x<=1)).all())

    def test_threshold_selection_does_not_reward_empty_predictions(self):
        # Three negatives must not outweigh the single positive target.
        y=np.array([[[1,0]],[[0,0]],[[0,0]],[[0,0]]],dtype=np.uint8)
        p=np.array([[[.3,.2]],[[.2,.2]],[[.2,.2]],[[.2,.2]]])
        chosen=threshold_search(p,y)
        self.assertEqual(chosen['threshold'],.3)
        self.assertEqual(chosen['metrics']['macro_nonempty']['dice'],1)
        json.dumps(chosen)  # Training and evaluation must persist NumPy counts as JSON.


if __name__=='__main__':
    unittest.main()
