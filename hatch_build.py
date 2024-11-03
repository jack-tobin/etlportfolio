from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from Cython.Build import cythonize
from distutils.core import Extension, Distribution
from distutils.command.build_ext import build_ext
import numpy as np
import glob
import os

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        include_dirs = [np.get_include()]

        pyx_files = glob.glob("etlportfolio/optimized/*.pyx")

        extensions = []
        for pyx_file in pyx_files:
            module_path = os.path.splitext(pyx_file)[0].replace("/", ".")
            extensions.append(
                Extension(
                    module_path,
                    [pyx_file],
                    include_dirs=include_dirs,
                ),
            )

        ext_modules = cythonize(
            extensions,
            compiler_directives={
                'language_level': "3",
                'boundscheck': False,
                'wraparound': False,
                'initializedcheck': False,
            }
        )

        build_data['extensions'] = ext_modules

    def finalize(self, version, build_data, artifact_path):
        """Build the extensions locally after the build process."""
        if 'extensions' not in build_data:
            return

        # Create a Distribution instance
        dist = Distribution({
            'name': 'etlportfolio',
            'ext_modules': build_data['extensions'],
        })

        # Create and configure build_ext command
        cmd = build_ext(dist)
        cmd.inplace = True
        cmd.extensions = build_data['extensions']

        # Build in place
        cmd.ensure_finalized()
        cmd.run()
