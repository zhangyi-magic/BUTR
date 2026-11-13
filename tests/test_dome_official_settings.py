import unittest

import torch

from src.core.yaml_utils import load_config
from src.zoo.models.dome_decoder import DomeTransformer


class DomeOfficialSettingsTests(unittest.TestCase):
    def test_dome_uses_shared_tiny_object_anchor_size(self):
        model = DomeTransformer(
            hidden_dim=32, feat_channels=[32], feat_strides=[8],
            num_levels=1, num_layers=1, num_points=1, num_denoising=0,
        )
        anchors, _ = model._generate_anchors([[2, 3]])
        self.assertTrue(torch.allclose(anchors.sigmoid()[..., 2:], torch.full((1, 6, 2), 0.0125)))

    def test_dome_configs_select_official_recipe_without_affecting_butr(self):
        original_butr = load_config('configs/models/BUTR-L-AITOD.yml', {})
        for scale in 'SML':
            for dataset in ('AITOD', 'VisDrone'):
                cfg = load_config(f'configs/models/Dome-{scale}-{dataset}.yml', {})
                self.assertFalse(cfg['use_amp'])
                self.assertEqual(cfg['criterion'], 'DomeCriterion')
                self.assertTrue(cfg['DomeCriterion']['use_defe_loss'])
                self.assertEqual(cfg['train_dataloader']['total_batch_size'], 16)
                self.assertEqual(cfg['val_dataloader']['total_batch_size'], 8)
                self.assertNotIn('anchor_grid_size', cfg['DomeTransformer'])
                self.assertEqual(cfg['HybridEncoder']['mwas_window_mode'], 'window_size')
        butr = load_config('configs/models/BUTR-L-AITOD.yml', {})
        self.assertEqual(butr, original_butr)
        self.assertFalse(butr['DomeCriterion']['use_defe_loss'])
        self.assertEqual(butr['train_dataloader']['total_batch_size'], 16)
        self.assertNotIn('anchor_grid_size', butr['DomeTransformer'])

    def test_defe_supervision_is_enabled_only_for_dome_configs(self):
        from pathlib import Path
        from src.core import YAMLConfig
        from src.zoo.models.dome_criterion import DomeCriterion

        paths = list(Path('configs/models').glob('*.yml'))
        paths += list(Path('configs/models/grain').glob('*.yml'))
        for path in paths:
            with self.subTest(config=str(path)):
                cfg = load_config(str(path), {})
                self.assertEqual(cfg['criterion'], 'DomeCriterion')
                self.assertEqual(cfg['DomeCriterion']['use_defe_loss'], path.name.startswith('Dome-'))
                loss = YAMLConfig(str(path)).criterion
                self.assertIsInstance(loss, DomeCriterion)
                self.assertEqual(loss.use_defe_loss, path.name.startswith('Dome-'))


if __name__ == '__main__':
    unittest.main()
