# CHANGELOG

<!-- version list -->

## v1.4.6 (2026-06-03)

### Bug Fixes

- Changed python version dependency to be >= 3.12 version
  ([`8a96a03`](https://github.com/sisinflab/warprec/commit/8a96a0324475cd46f6760663b287cdb0660d83d9))

- CL4SRec model operation vectorization
  ([`9af00e8`](https://github.com/sisinflab/warprec/commit/9af00e88bb493697c1ff78c62d9a61ce07404bd8))

- ESASRec model operation vectorization
  ([`65961b3`](https://github.com/sisinflab/warprec/commit/65961b3887ea78da86172dad069e97db00bb3232))

- LinRec model Q and K scaling factor during normalization
  ([`d79c7ec`](https://github.com/sisinflab/warprec/commit/d79c7ecf5166721c9ce57e35defe6695b1f9b442))

- PopDCL model now applies normalization correctly during predict
  ([`8e98cc3`](https://github.com/sisinflab/warprec/commit/8e98cc3ae81e54d50e576f0341cb3a52db33c814))

### Documentation

- Removed unused hyperparameter from CORE configuration example
  ([`c4888be`](https://github.com/sisinflab/warprec/commit/c4888bec613cda55840da3bf475a2e849501ecc9))


## v1.4.5 (2026-05-22)

### Bug Fixes

- Added address as general configuration parameter
  ([`bac5b93`](https://github.com/sisinflab/warprec/commit/bac5b93aa66d50161aba3ba5a62fd8aa54cbac15))

- Evaluation dataloaders will accept and correctly cache kwargs
  ([`1fa6e13`](https://github.com/sisinflab/warprec/commit/1fa6e132ecdb5913e0a1f189d719b127c569f234))

- Model extended name depends also on original model name to ensure that models with same parameters
  generate the same name extension
  ([`c364ddb`](https://github.com/sisinflab/warprec/commit/c364ddb810e8e3ae91da2725414941c8b152dd74))

### Documentation

- Added ray address to documentation
  ([`b4494a2`](https://github.com/sisinflab/warprec/commit/b4494a2b242543407f1666110b67d3e96d38e7bb))


## v1.4.4 (2026-05-20)

### Bug Fixes

- Fixed bug with Timestamp Slicing splitting strategy which corrupted dataset dimensions
  ([`81aa8b6`](https://github.com/sisinflab/warprec/commit/81aa8b683b4f95fa3d2511ef1727a91a937a824c))

- WarpRec will now log internal alignment filtering on evaluation sets
  ([`f1e85ca`](https://github.com/sisinflab/warprec/commit/f1e85ca426204c4fd67f844c48aa12f6260d9a36))


## v1.4.3 (2026-05-13)

### Bug Fixes

- Model names will be created using coolname based on their parameter hash
  ([`b424f7a`](https://github.com/sisinflab/warprec/commit/b424f7a33deb41472adaeca07a3cb02fff6edbad))

- Trial name during training will be derived from the new naming convention
  ([`86f14da`](https://github.com/sisinflab/warprec/commit/86f14da7c4312fc564fe0198b9ba949d7ade0575))

### Chores

- Added coolname dependency
  ([`d606015`](https://github.com/sisinflab/warprec/commit/d606015f4a057a713cf3b9a3e36d4d013fd4914d))


## v1.4.2 (2026-05-06)

### Bug Fixes

- Added concurrent trial estimation and placement group compute during trial allocation
  ([`3dabb4b`](https://github.com/sisinflab/warprec/commit/3dabb4ba737c36bdfe0231a8f40873f463c588e8))

- Custom Ray Train trial ID to prevent race condition during trial allocation
  ([`8627f52`](https://github.com/sisinflab/warprec/commit/8627f5267f82541599704cf3920a833bf873baf7))

- Removed placement group strategy and delegating resource management to Ray Train
  ([`66c3520`](https://github.com/sisinflab/warprec/commit/66c35201bc73d72949228ac58915dabdb1a5dbde))

### Chores

- Added max_concurrent_trial configuration
  ([`7d1e548`](https://github.com/sisinflab/warprec/commit/7d1e548d58494eadae6d106aae19554c434cffa5))


## v1.4.1 (2026-05-05)

### Bug Fixes

- Added warning message in the swarm pipeline
  ([`750e321`](https://github.com/sisinflab/warprec/commit/750e321bdc349a27cee0f7cca3774f8cbb207a75))

- Corrected PAAC model loss computation
  ([`074d11a`](https://github.com/sisinflab/warprec/commit/074d11aacc4c909400ccb5ce7ce52533df4b7792))

### Documentation

- Added references to the new estimate pipeline
  ([`fcad7b1`](https://github.com/sisinflab/warprec/commit/fcad7b104a66ab9b5d7e03be09976855ee1edac7))


## v1.4.0 (2026-05-04)

### Bug Fixes

- Fixed PAAC model implementation to correctly reflect original paper formulation
  ([`057b727`](https://github.com/sisinflab/warprec/commit/057b727d37cdc621562e4cc094233ddb91c15f21))

- Removed max trial constraint
  ([`d0be3be`](https://github.com/sisinflab/warprec/commit/d0be3beb719fbb5cc6f4071ea599a467fcf4f97b))

- RP3Beta model initialization
  ([`b1e002e`](https://github.com/sisinflab/warprec/commit/b1e002e8d3510cb9f1c723618e21ed84c7cbe16e))

### Documentation

- Added new models to the documentation
  ([`bbead3c`](https://github.com/sisinflab/warprec/commit/bbead3cf89d9a2424979fa1c922d81d4ed10649f))

### Features

- Added bsarec model implementation
  ([`af91531`](https://github.com/sisinflab/warprec/commit/af915318eaf9b4fb73ded1d0b71d22471b2fcdf3))

- Added cl4srec model implementation
  ([`1e26bee`](https://github.com/sisinflab/warprec/commit/1e26bee42c01d02ccd0d5d703dccbdd3a8c90293))

- Added duorec model implementation
  ([`21a7345`](https://github.com/sisinflab/warprec/commit/21a7345bfed7744eeff54ebb609d95ce61dd4e55))

- Added esasrec model implementation
  ([`2390ce9`](https://github.com/sisinflab/warprec/commit/2390ce97f21dd7ff519d1e11d9a275a020609adc))

- Added iALS model implementation
  ([`29b583d`](https://github.com/sisinflab/warprec/commit/29b583d3916a9e890321bf2a31204a84009b74ec))

- Added iALS2008 model implementation
  ([`11b5eed`](https://github.com/sisinflab/warprec/commit/11b5eed8eda9888df2e05be5b443a01f2d86fa53))

- New dataloader for same target sequential data
  ([`2da35bf`](https://github.com/sisinflab/warprec/commit/2da35bf1e640830a705bc6e5aeec10a4f251cd42))


## v1.3.2 (2026-04-29)

### Bug Fixes

- MACRMF parameter validation fixed
  ([`0f98503`](https://github.com/sisinflab/warprec/commit/0f98503b402a2fd6c04d595c251041be1cb6d14f))

- Temporal Decay KNN models parameter validation fixed
  ([`cbea62b`](https://github.com/sisinflab/warprec/commit/cbea62bcc9f43d8e3507ef37e62c894c59c2fc8a))

- Train and Swarm pipeline will assign a fallback value to number of GPUs for cuda models with no
  specified number of cuda devices per trial
  ([`5d2b4d7`](https://github.com/sisinflab/warprec/commit/5d2b4d7f4cf50bf489621504e136abb6894c96ba))

### Documentation

- Updated model taxonomy and counting
  ([`94a2c0f`](https://github.com/sisinflab/warprec/commit/94a2c0fa9f30bff433f549281c62e67ef54363cf))


## v1.3.1 (2026-04-23)

### Bug Fixes

- Full evaluation will now iterate over only the test set, not the full set of users and items
  ([`505853a`](https://github.com/sisinflab/warprec/commit/505853ae6610a7f33721197bb4e37000ff9b3dc7))


## v1.3.0 (2026-04-22)

### Bug Fixes

- Correct ASHA initialization
  ([`a9669ed`](https://github.com/sisinflab/warprec/commit/a9669ed9067989e8a56edb1a189b997ad727a3c5))

- Correctly initialize optimization metric for bayesian strategies
  ([`4b9299e`](https://github.com/sisinflab/warprec/commit/4b9299eb2119dc7955577851ec95ee10861b5617))

- Correctly loading custom modules on remote worker nodes
  ([`7e0b168`](https://github.com/sisinflab/warprec/commit/7e0b1687e119dcfd1042eb4367e742c74ccf0e2d))

- Correctly set reproducibility after final evaluation model de-serialization
  ([`f855350`](https://github.com/sisinflab/warprec/commit/f8553505c7f5ed899fba26f3a6da6dd9f2847598))

- Metrics will not be validated with config, allowing custom metrics to be correctly imported
  ([`d2867a0`](https://github.com/sisinflab/warprec/commit/d2867a07e511ac5e597de1194a29d5782d584a51))

- Serialization error of Pop model
  ([`64fb583`](https://github.com/sisinflab/warprec/commit/64fb583bb49697630ce319ebdd1d2c2f3a494658))

- Tuning process will now account for driver function resources during CPU-bound optimizations
  ([`f2c6938`](https://github.com/sisinflab/warprec/commit/f2c693873caa3a20434ec391051c6d5521f26f8d))

### Documentation

- Added ASHA scheduler example usage
  ([`b514b11`](https://github.com/sisinflab/warprec/commit/b514b11c31b3bff9d919ad3b0cf5fe75cc5d349b))

- Added context label in the configuration setting
  ([`134ef19`](https://github.com/sisinflab/warprec/commit/134ef197110a3093f8c0f905124349946b51bc49))

- Updated documentation with new models
  ([`290127e`](https://github.com/sisinflab/warprec/commit/290127e137ef12226e26cc53b2b90a6a62858428))

### Features

- Added ItemKNN-TD model implementation
  ([`6c01021`](https://github.com/sisinflab/warprec/commit/6c01021b91e9e66654aa5fc5a4a0edb2557bbe60))

- Added MACRGCN model implementation
  ([`98bf4e8`](https://github.com/sisinflab/warprec/commit/98bf4e84643f3e65b1524f0ac645fee4713d3d7e))

- Added MACRMF model implementation
  ([`91d162a`](https://github.com/sisinflab/warprec/commit/91d162a3bc92647a493168170c48e3ce3b9681f6))

- Added PAAC model implementation
  ([`c2f5c15`](https://github.com/sisinflab/warprec/commit/c2f5c159e3f36e1574f80032ecade643491f02a9))

- Added PopDCL model implementation
  ([`0a182a0`](https://github.com/sisinflab/warprec/commit/0a182a07c043988c44e72125ecb1f611a2bcd882))

- Added RecDCL model implementation
  ([`738404a`](https://github.com/sisinflab/warprec/commit/738404ac0e4cecd4881d0643bb406c82bb06f6b0))

- Added SimGCL model implementation
  ([`abba348`](https://github.com/sisinflab/warprec/commit/abba34881ec2a129fc16f14456775430ec3bb97d))

- Added SimRec model implementation
  ([`37a7257`](https://github.com/sisinflab/warprec/commit/37a72579868a2541d68cd9bde713509254c45645))

- Added STAN model implementation
  ([`79cc29a`](https://github.com/sisinflab/warprec/commit/79cc29a54c8bbe048490a44e9f01bf9c58b9b774))

- Added UserKNN-TD model implementation
  ([`3034ad5`](https://github.com/sisinflab/warprec/commit/3034ad5b6c749a0a7c98c254eba79d50c31090c8))

- Codecarbon local logging will be saved in WarpRec experiment directory
  ([`6d8ffff`](https://github.com/sisinflab/warprec/commit/6d8ffff610e9af8f305a7af7281abc552507d038))


## v1.2.2 (2026-04-15)

### Bug Fixes

- Move remote evaluated metrics to cpu to avoid CUDA errors on driver node
  ([`5361e3a`](https://github.com/sisinflab/warprec/commit/5361e3a2a00462de702182478735706106523d24))


## v1.2.1 (2026-04-14)

### Bug Fixes

- Correct invalidation of graph recommenders local cache
  ([`c940e33`](https://github.com/sisinflab/warprec/commit/c940e33e0eecd9706fba409494b2ec01f243fd67))

### Documentation

- Added llms-txt plugin
  ([`dda4a22`](https://github.com/sisinflab/warprec/commit/dda4a22f6f1fa6c02455e13a326a8f8ff6dc6f08))


## v1.2.0 (2026-04-09)

### Bug Fixes

- Reduces resources requested by the driver function
  ([`f863b48`](https://github.com/sisinflab/warprec/commit/f863b48c0bfbf69847305a7c75d27a223321f6e8))

### Documentation

- Added data preparation resources configuration
  ([`8b2215f`](https://github.com/sisinflab/warprec/commit/8b2215fb766c433af8cf9ce1c194ef41c9e23b1c))

- Added documentation of swarm pipeline
  ([`e0bf466`](https://github.com/sisinflab/warprec/commit/e0bf4660a18970fe96c37356df8de67ba43f3f97))

### Features

- Added new pipeline "swarm"
  ([`c9abea6`](https://github.com/sisinflab/warprec/commit/c9abea6e04d7f34774285d984fae897d9be413e5))

- Added remote functions to offload execution to Ray cluster
  ([`c379638`](https://github.com/sisinflab/warprec/commit/c379638db859a0826acb9af839592b35a6f4a6a0))


## v1.1.3 (2026-04-03)

### Bug Fixes

- Correct metric reporting for non iterative recommenders
  ([`e26c837`](https://github.com/sisinflab/warprec/commit/e26c8379f5298039660202871915e0f381aede01))

- Metric report during training will accept non-tensors metrics
  ([`83d8c98`](https://github.com/sisinflab/warprec/commit/83d8c98a7e7ed0da819c727aaae6673dd9c2ab6f))


## v1.1.2 (2026-04-03)

### Bug Fixes

- Correctly create the instance of the best model after the training is complete
  ([`79cc2b6`](https://github.com/sisinflab/warprec/commit/79cc2b6d0ac780d4d6be372f8f5dc9d1bad19a36))


## v1.1.1 (2026-04-02)

### Bug Fixes

- Added label selector to optimization configuration
  ([`93172af`](https://github.com/sisinflab/warprec/commit/93172af4a2c6b486d92fe7ca5f0a02019c6cf8ef))

### Chores

- Linting
  ([`919f925`](https://github.com/sisinflab/warprec/commit/919f92503116a1c6d38eef6241ec2bfc5af12267))

- Updated lock file
  ([`ef7babb`](https://github.com/sisinflab/warprec/commit/ef7babb85e3d1b422fbc69178466ed57d6c844e4))

- Updated Ray dependency minimum version
  ([`e3a9e4d`](https://github.com/sisinflab/warprec/commit/e3a9e4d59286a6153a603df29d4db645a07c4ccf))

### Documentation

- Added label selector to documentation
  ([`a41c69b`](https://github.com/sisinflab/warprec/commit/a41c69b06f69f84e915834a70d8ef355f55bffc1))

- Fixed api reference
  ([`d352aa0`](https://github.com/sisinflab/warprec/commit/d352aa0c448da5217a2f0d57e77a5bfdb19cda43))


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
