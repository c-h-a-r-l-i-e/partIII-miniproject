"""
Temporary script to repair some issues previously caused. Downsizes videos to the correct fps values.
"""

import pandas as pd
import os




fps_list = [5, 15]
res_list = [144, 360, 720]
emotions = range(1, 9, 1)
emo_intensities = [1,2]
statements = [1,2]
reps = [1,2]
actors = range(1, 25, 1)

create_csv = False

data = []
for emotion in emotions:
    for emo_intensity in emo_intensities:
        for statement in statements:
            for rep in reps:
                for actor in actors:
                    for res in res_list:
                        fname = "../data/dataset/{:02d}-{:02d}-{:02d}-{:02d}-{:02d}.mp4".format(emotion, emo_intensity, statement, rep, actor, 30, res)
                        for fps in fps_list:
                            new_fname = "../data/dataset/{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{}-{}.mp4".format(emotion, emo_intensity, statement, rep, actor, fps, res)
                            os.system("ffmpeg -i {} -filter:v fps=fps={} {}".format(fname, fps, new_fname))
                            data.append(
                                {
                                    "actor" : actor,
                                    "fps" : fps,
                                    "res" : res,
                                    "emotion" : emotion,
                                    "emotion intensity" : emo_intensity,
                                    "statement" : statement,
                                    "repitition" : rep,
                                    "video file" : new_fname
                                }
                            )

if create_csv:
    df = pd.DataFrame(data)
    df.to_csv("dataset.csv", index=False)

