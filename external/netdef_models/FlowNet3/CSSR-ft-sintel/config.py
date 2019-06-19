import netdef_slim as nd
import os

nd.evo_manager.set_training_dir(os.path.join(os.path.dirname(__file__), 'training'))

schedule = nd.FixedStepSchedule('S_custom', max_iter=250000, steps=[5000, 10000, 20000], base_lr=1e-05)
nd.add_evo(nd.Evolution('sintel_mixture', [], schedule))
