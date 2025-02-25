project('my_project', 'c',  default_options: ['buildtype=debug', 'optimization=0', 'warning_level=3'])

sources = [
    'c_omp.c'
]

deps = [dependency('openmp')]

executable('sor', sources, dependencies: deps)