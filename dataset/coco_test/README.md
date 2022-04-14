MS COCO Examples
================

This folder holds an exemplary excerpt from the
[2017 MS COCO dataset](https://cocodataset.org/) annotations.

For testing, put the images mentioned in the annotation files into the folders
``images/train2017`` respectively ``images/val2017``.  
This can, e.g., be achieved using the following commands in a bash:
```bash
for json_file in $(ls annotations/*.json); do
	for url in $(grep coco_url $json_file | sed "s%^.*\\(http.*jpg\\).*$%\\1%"); do
		curr_dir=images/$(echo $json_file | sed "s%person_keypoints_\\(.*\\)\\.json%\\1%g");
		mkdir -p $curr_dir;
		wget $url -O $curr_dir/$(basename "$url");
	done;
done;
```

The annotation files in the [annotations folder](./annotations) are excerpts from the original [MS COCO annotations](https://cocodataset.org/#download).
They are licensed under the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Find the attributions in the corresponding ``ATTRIBUTIONS.md`` (same folder as the annotations).
