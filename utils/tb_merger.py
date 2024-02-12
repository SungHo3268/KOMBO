from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import os


# Set the paths of the two event files to be merged
file1 = '/data2/user13/workspace/BTS_Tokenization/charformer/logs/charformer-base/pretraining/en_utf-8_144b_4s_1024seq_65536/tb/events.out.tfevents.1683027408.devbox.310504.0'
file2 = '/data2/user13/workspace/BTS_Tokenization/charformer/logs/charformer-base/pretraining/en_utf-8_144b_4s_1024seq_65536/tb/events.out.tfevents.1683636083.devbox.2161114.0'

# Create event accumulators for the two files
ea1 = EventAccumulator(file1)
ea1.Reload()
ea2 = EventAccumulator(file2)
ea2.Reload()

# Create a new event file to merge the data into
merged_file = '/data2/user13/workspace/BTS_Tokenization/charformer/logs/charformer-base/pretraining/en_utf-8_144b_4s_1024seq_65536/tb/'
writer = SummaryWriter(merged_file)

# Get the tags from the first event file and write them to the new file
tags = ea1.Tags()['scalars']
with writer:
    for tag in tags:
        scalar_events = ea1.Scalars(tag) + ea2.Scalars(tag)
        for event in scalar_events:
            writer.add_scalar(tag, event.value, global_step=event.step, walltime=event.wall_time)


os.remove(path=file1)
os.remove(path=file2)
