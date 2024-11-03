import os
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from Cython.Build import cythonize
from distutils.core import Extension, Distribution
from distutils.command.build_ext import build_ext
import numpy as np


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        include_dirs = [np.get_include()]

        extensions = [
            Extension(
                "etlportfolio.optimized.risk_criteria",
                ["etlportfolio/optimized/risk_criteria.pyx"],
                include_dirs=include_dirs,
            ),
        ]

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

        build_dir = os.path.dirname(artifact_path)

        # Create a Distribution instance
        dist = Distribution({
            'name': 'etlportfolio',
            'ext_modules': build_data['extensions'],
        })

        # Create and configure build_ext command
        cmd = build_ext(dist)
        cmd.inplace = False
        cmd.force = True
        cmd.extensions = build_data['extensions']
        cmd.build_lib = build_dir
        cmd.build_temp = os.path.join(build_dir, 'temp')

        os.makedirs(cmd.build_temp, exist_ok=True)

        # Build in place
        cmd.ensure_finalized()
        cmd.run()
