"""Bounded semantic overlay export from already-computed masks."""
from pathlib import Path
import numpy as np
from PIL import Image


def save_overlay(output,scan,image_index,view_index,rendered,labels,classes):
    h,w = labels.shape
    background = np.array(Image.fromarray(rendered).resize((w,h),Image.Resampling.LANCZOS))
    palette = np.array([[int(c['color'][i:i+2],16) for i in (1,3,5)] for c in classes],np.uint8)
    accepted = labels >= 0
    colored = background.copy()
    colored[accepted] = np.rint(.55*background[accepted]+.45*palette[labels[accepted]]).astype(np.uint8)
    directory = Path(output)/'diagnostics'/f'scan_{scan:04d}'
    directory.mkdir(parents=True,exist_ok=True)
    stem = f'image_{image_index:04d}_view_{view_index:02d}'
    Image.fromarray(colored).save(directory/(stem+'_classes.jpg'),quality=92)
    Image.fromarray(background).save(directory/(stem+'_source.jpg'),quality=92)
