# Test (unit test)
#PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=moco_ego4d_unit_test data.train_filelist=datasets/ego4d_tiny.txt environment.ngpu=8 environment.world_size=2 optim.epochs=2

PYTHONPATH=. python main_moco_clip.py data.type=imagenet model.moco_k=256 optim.batch_size=128 environment.slurm=True logging.name=moco_clip_imnet environment.ngpu=8 environment.world_size=1

# Train on 100K frames
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=ego4d_100k data.train_filelist=datasets/ego4d_100k.txt environment.ngpu=8 environment.world_size=2

# Train on approx 5 million frames (r3m dataset)
#PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=ego4d_5m data.train_filelist=datasets/ego4d_5m.txt environment.ngpu=8 environment.world_size=2
