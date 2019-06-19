import netdef_slim as nd
import os

nd.evo_manager.set_training_dir(os.path.join(os.path.dirname(__file__), 'training'))

nd.add_evo(nd.Evolution('SDMixture_no_hom', [], 'S_refinement'))
