import glob
import shutil
import subprocess
import random
from tqdm import tqdm

from sac.model.audacity_label import AudacityLabel
from sac.util import Util


print " - Creating ground truth..."

files = glob.glob("./audio/*.wav")

music_wavs = []
speech_wavs = []
for f in files:
    gt = f.split('_-_')[-1]
    if "no-music" in gt:
        speech_wavs.append(f)
    else:
        music_wavs.append(f)

all_files_dict = {}

for f in speech_wavs:
    all_files_dict[f] = "s"

for f in music_wavs:
    all_files_dict[f] = "m"

random.seed(1111)
all_files_random_keys = random.sample(all_files_dict.keys(), len(all_files_dict.keys()))

last_seconds = 0
files_to_concatenate = []

labels = []
for v in tqdm(all_files_random_keys):
    duration = float(subprocess.check_output(["soxi", "-D", v]).strip())
    segment_start_time = last_seconds
    segment_end_time = last_seconds + duration
    last_seconds += duration
    labels.append(AudacityLabel(segment_start_time, segment_end_time, all_files_dict[v]))
    files_to_concatenate.append(v)
print len(files_to_concatenate)
audacity_labels = Util.combine_adjacent_labels_of_the_same_class(labels)
Util.write_audacity_labels(audacity_labels, "audio_0_combined.txt")

print " - Concatenating wav files..."
for i, fname in tqdm(enumerate(files_to_concatenate)):
    if i == 0:
        subprocess.check_output(["cp", fname, "audio_0_combined.wav"])
        continue
    command = ["sox", "audio_0_combined.wav", fname, "temp.wav"]
    subprocess.check_output(command)
    subprocess.check_output(["cp", "temp.wav", "audio_0_combined.wav"])
