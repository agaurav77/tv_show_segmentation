**Goal**: trying to do tv show segmentation on unusually large digital silos to divide it into meaningful segments

**Dependencies**: moviepy, numpy, imageio

**Submodules**: darknet, pyAudioAnalysis

> Also use [marsyas](http://marsyasweb.appspot.com/download/data_sets/) datasets for HMM model generation

**Supported cues**

* _predict_yolo(frame)

> returns dict of with keys as items detected, values as probabilities of detection, uses tiny-yolo 

* detect_black_frame(frame, threshold=0.95)

> returns True if frame is at least threshold percent dark

* match_templates(files, frame)

> match all template image files (logo files say) to the given frame; returns True if the max number of hits was greater than or equal to 6, uses SIFT and FLANN

* audio_classify(audio_clip)

> runs HMM classification to identify segments of given audio_clip