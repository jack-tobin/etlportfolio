from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from Cython.Build import cythonize
import numpy as np

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        extensions = cythonize("etlportfolio/optimized/*.pyx")
        for ext in extensions:
            ext.include_dirs.append(np.get_include())
        build_data['extensions'] = extensions
