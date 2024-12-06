---
sidebar_position: 4
---

# Refine Motion (Post-process noisy tracking motion)

The input marker data may be noisy or contain missing data in a few frames. We can run a smoothing step on the entire sequence to fill in gaps and smooth out noise.

Use a config file:

<OssOnly>
```
pixi run refine_motion -c refine_motion.config
```
</OssOnly>

<FbInternalOnly>
```
buck run @<mode> refine_motion -- -c refine_motion.config
```
</FbInternalOnly>

Use cli argument to overwrite config values:

<OssOnly>
```
pixi run refine_motion -c refine_motion.config -i track.glb -o refined.glb --smoothing 2.5
```
</OssOnly>

<FbInternalOnly>
```
buck run @<mode> refine_motion -- -c refine_motion.config -i track.glb -o refined.glb --smoothing 2.5
```
</FbInternalOnly>
