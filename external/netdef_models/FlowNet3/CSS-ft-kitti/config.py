import netdef_slim as nd
import os
schedule = nd.FixedStepSchedule('S_custom', max_iter=200000, steps=[50000, 90000, 150000], base_lr=1e-05)
nd.evo_manager.set_training_dir(os.path.join(os.path.dirname(__file__), 'training'))
nd.add_evo(nd.Evolution('kitti.train', [], schedule))
