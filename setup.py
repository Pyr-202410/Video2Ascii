from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(
    name='V2A',
    ext_modules=cythonize("Video2Ascii.pyx"),
    include_dirs=[np.get_include()],  # 添加这一行
    requires=['Cython', 'numpy', 'cv2', 'Pillow', 'tqdm']
)