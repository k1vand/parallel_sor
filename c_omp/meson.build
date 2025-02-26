project('my_project', 'c',  default_options: ['buildtype=debug', 'optimization=0', 'warning_level=3'])

sources = [
    'c_omp.c'
]

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)

deps = [dependency('openmp'), m_dep]

executable('sor', sources, dependencies: deps)