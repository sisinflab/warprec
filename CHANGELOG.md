# CHANGELOG

<!-- version list -->

## v1.1.0 (2026-04-02)

### Bug Fixes

- Added validation_step hook to base model interface
  ([`a10951f`](https://github.com/sisinflab/warprec/commit/a10951fcc63a45710bad0d9e9a636622e9b83afc))

- Correct initialization of Evaluator during special conditions
  ([`f273fd5`](https://github.com/sisinflab/warprec/commit/f273fd56b9609771098a60c72e49314db0f44dd4))

- Explicit declaration of different checkpoint configuration in Trainer
  ([`cca8159`](https://github.com/sisinflab/warprec/commit/cca815961684e8f4ad03f6ee71b5c8ef571f7ebc))

- Fixed checkpoint retrieval and result logging
  ([`507e234`](https://github.com/sisinflab/warprec/commit/507e23432ec3d2ac782301cf786879cc8cce4d84))

- Moved early stopping inside integration callback
  ([`ba2fa6b`](https://github.com/sisinflab/warprec/commit/ba2fa6b7c9eb6778f75ca83c2dff0af672400d39))

- Moved optimizer initialization at model level
  ([`7798187`](https://github.com/sisinflab/warprec/commit/7798187733d672194a16337b69cf22318bdd7614))

### Chores

- Added mkdocs to dev dependencies
  ([`77c8e5d`](https://github.com/sisinflab/warprec/commit/77c8e5d69d81769397bdbefe5c7a76845da741d0))

- Added optimizer configuration to model config
  ([`57257fe`](https://github.com/sisinflab/warprec/commit/57257fe1f733e875d1e68fac6db2ec6a82e06aaa))

- Added optimizer registry
  ([`29820c5`](https://github.com/sisinflab/warprec/commit/29820c54f7f38f16f55d66e069a28c2d06ebae1d))

- Added PyTorch Lightning dependency
  ([`279681e`](https://github.com/sisinflab/warprec/commit/279681e3e97e690e680964355ebbb1359925fea5))

- Added to model interface method to set optimization parameters
  ([`b5d4181`](https://github.com/sisinflab/warprec/commit/b5d418181af440595465af8859d8445dbd4e2b04))

- Removed deprecated import
  ([`c8afd83`](https://github.com/sisinflab/warprec/commit/c8afd83034977d7f106867de0468b90e89dad805))

- Removed Ray environment variable from train pipeline
  ([`8b12a9a`](https://github.com/sisinflab/warprec/commit/8b12a9a60e0ec81c69057dbebba2812c91e0bd38))

- Updated lock hashing
  ([`0336da0`](https://github.com/sisinflab/warprec/commit/0336da018e4e4e86c7ae2101669684d5ab9a29f1))

### Code Style

- Changed loss logging naming
  ([`6757399`](https://github.com/sisinflab/warprec/commit/6757399e74f0060b3eb44203698fb1e8b37d543f))

### Documentation

- Added poetry to official doc
  ([`fe96beb`](https://github.com/sisinflab/warprec/commit/fe96beb54539b7b12bb27edbfc630d2c76edcd70))

- Added poetry to README
  ([`cf269a5`](https://github.com/sisinflab/warprec/commit/cf269a55d447ff82ccf70690698e99e4f75d6f59))

- Added reference to optimizer configuration
  ([`d9f3b45`](https://github.com/sisinflab/warprec/commit/d9f3b45526d3e412345ff74955db886e544627bf))

- Cleaned README
  ([`ed93997`](https://github.com/sisinflab/warprec/commit/ed939975b7c1badaadaa6419e2dd372271552099))

- Fixed bash section in poetry installation guide
  ([`107e030`](https://github.com/sisinflab/warprec/commit/107e030dc49521052b35c98bd92ea808f99179d6))

### Features

- Added hook on_save_checkpoint to recommender model interface
  ([`9879fce`](https://github.com/sisinflab/warprec/commit/9879fce341508e7773b2474fc83a0b1ed6483e13))

- Added optimizer customization to main pipelines
  ([`bf46649`](https://github.com/sisinflab/warprec/commit/bf46649eac6f974883a7b6d8bb6003095f3771e5))

- Added PyTorch Lightning callback implementation on validation
  ([`0cfadaa`](https://github.com/sisinflab/warprec/commit/0cfadaa806aad13e27ce3884c72647f86809de17))

- Added PyTorch Lightning model integration
  ([`50afd52`](https://github.com/sisinflab/warprec/commit/50afd527ef34787b0ce8a014a0cb8e9484fa6155))

- Added WarpRec and Lightning integration callback
  ([`086bb35`](https://github.com/sisinflab/warprec/commit/086bb35a5262cd37eb645b03754f30223a086bb6))

- Unified objective function logic (CPU, GPU and DDP) and integrated Lightning Trainer
  ([`beee73e`](https://github.com/sisinflab/warprec/commit/beee73e6984300074d86975f806026cf7740aa00))

- Updated design pipeline with standard Lightning Trainer
  ([`826b9e3`](https://github.com/sisinflab/warprec/commit/826b9e37e5175bbe1e012ccceb51084f88218091))

- Updated train pipeline with standard Lightning Trainer
  ([`5396d2a`](https://github.com/sisinflab/warprec/commit/5396d2a7ea11cfe2d3d78c060b3d198ce76611cd))

### Refactoring

- Implemented back the loss, best score and memory logging inside Tuner
  ([`966e619`](https://github.com/sisinflab/warprec/commit/966e61981ee6140e7a124f2d27ad651482c2778d))

- Moved evaluation data loading logic at dataset level
  ([`0fe3c01`](https://github.com/sisinflab/warprec/commit/0fe3c014dff80258e350eef02fce0f2b5de04d3b))

- Removed loops utility
  ([`eee70ad`](https://github.com/sisinflab/warprec/commit/eee70adb4017408dcb5c56992ae6e792c54d70bc))

- Updated model interfaces
  ([`d6fba58`](https://github.com/sisinflab/warprec/commit/d6fba58bcb7eeaa6d9ce2bc4737a8abea672b65e))

- Updated Trainer with new Ray Train + Ray Tune + Lightning integration
  ([`070be1f`](https://github.com/sisinflab/warprec/commit/070be1f9da648b52635c642bdf24d10fa842549f))

- Using RayTrainReportCallback to correctly report metrics and checkpoints
  ([`5ce38f5`](https://github.com/sisinflab/warprec/commit/5ce38f5c5704da6705c34ebf9be5f9753017e1a6))


## v1.0.1 (2026-03-20)

### Bug Fixes

- **entrypoint**: Added main warprec entrypoint
  ([`94b4cf7`](https://github.com/sisinflab/warprec/commit/94b4cf771cbf2771ed612cb5e719545479df1a9b))

### Code Style

- Typo
  ([`829f16f`](https://github.com/sisinflab/warprec/commit/829f16f2b347484a23818fcedbaddbad4f877e44))

### Documentation

- Updated official docs installation guide
  ([`8aa316b`](https://github.com/sisinflab/warprec/commit/8aa316bdf2263219fa909485779109549cb2631c))

- Updated README
  ([`f2f1891`](https://github.com/sisinflab/warprec/commit/f2f189128e1e6bc7152ab017452481b888a1a8ed))

- Updated README installation guide
  ([`ad65a8c`](https://github.com/sisinflab/warprec/commit/ad65a8cad1496cef95ba45bb5c552b6e7aa354b2))


## v1.0.0 (2026-03-19)

- Initial Release
